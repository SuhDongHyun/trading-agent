import json
from decimal import Decimal

import pytest

from trading_agent.application.recommendation_schema import (
    RECOMMENDATION_RESPONSE_SCHEMA,
    parse_model_recommendation,
)
from trading_agent.domain.consensus import DisagreementReasonCode
from trading_agent.domain.errors import DomainError
from trading_agent.domain.models import (
    Action,
    ConfidenceAggregation,
    ConsensusBasis,
    ConsensusPolicyName,
)
from trading_agent.domain.recommendation import (
    ModelRecommendation,
    RecommendationConsensusPolicy,
    StockRecommendation,
)


def item(
    code: str = "005930",
    rating: Action = Action.BUY,
    confidence: float = 0.8,
    target: str | None = "120",
    stop: str | None = "90",
    score: str | None = None,
    expected: str | None = None,
    drivers: list[str] | None = None,
    risk_type: str | None = None,
    risk_note: str | None = None,
) -> StockRecommendation:
    return StockRecommendation(
        code=code,
        rating=rating,
        confidence=confidence,
        target_price=None if target is None else Decimal(target),
        invalidation_price=None if stop is None else Decimal(stop),
        reason=f"{rating.value} reason",
        score=None if score is None else Decimal(score),
        expected_excess_return_10d=None if expected is None else Decimal(expected),
        drivers=drivers or [],
        key_risk_type=risk_type,
        key_risk_note=risk_note,
    )


def opinion(items: list[StockRecommendation]) -> ModelRecommendation:
    return ModelRecommendation("fake", items, "summary", ["info"])


def test_matching_buy_keeps_lower_target_higher_stop_and_lower_confidence() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion(
            [
                item(
                    confidence=0.9,
                    target="130",
                    stop="88",
                    score="1.4",
                    expected="2",
                    drivers=["trend_up"],
                    risk_type="event_risk",
                    risk_note="실적 발표",
                )
            ]
        ),
        opinion(
            [
                item(
                    confidence=0.7,
                    target="120",
                    stop="92",
                    score="0.6",
                    expected="4",
                    drivers=["relative_strength"],
                    risk_type="macro_shock",
                    risk_note="환율 변동",
                )
            ]
        ),
    )

    merged = result.recommendations[0]
    assert merged.rating is Action.BUY
    assert merged.confidence == 0.7
    assert merged.target_price == Decimal("120")
    assert merged.invalidation_price == Decimal("92")
    assert merged.score == Decimal("1.0")
    assert merged.expected_excess_return_10d == Decimal("3")
    assert merged.evidence_by_model["claude"].drivers == ["trend_up"]
    assert merged.evidence_by_model["codex"].key_risk_type == "macro_shock"
    assert result.disagreements == []
    assert result.missing_info == ["info"]


def test_matching_sell_keeps_higher_target_and_lower_stop() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion([item(rating=Action.SELL, target="80", stop="115")]),
        opinion([item(rating=Action.SELL, target="85", stop="110")]),
    )

    merged = result.recommendations[0]
    assert merged.target_price == Decimal("85")
    assert merged.invalidation_price == Decimal("110")


def test_missing_price_on_either_side_is_not_invented() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion([item(target=None)]), opinion([item(target="120")])
    )

    assert result.recommendations[0].target_price is None


def test_hold_and_omitted_stock_agree_as_hold_without_record() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion([item(rating=Action.HOLD, target=None, stop=None)]), opinion([])
    )

    assert result.disagreements == []
    assert result.recommendations == []


def test_both_hold_is_kept_as_agreed_hold_opinion() -> None:
    hold = item(rating=Action.HOLD, target=None, stop=None)
    result = RecommendationConsensusPolicy.reconcile(opinion([hold]), opinion([hold]))

    assert [entry.rating for entry in result.recommendations] == [Action.HOLD]


@pytest.mark.parametrize(
    ("claude_items", "codex_items", "reason"),
    [
        ([item()], [item(rating=Action.SELL)], DisagreementReasonCode.ACTION_MISMATCH),
        ([item()], [], DisagreementReasonCode.ACTION_MISMATCH),
        ([item(), item()], [item()], DisagreementReasonCode.DUPLICATE_PROPOSAL),
    ],
)
def test_conflicts_are_recorded_and_not_adopted(
    claude_items: list[StockRecommendation],
    codex_items: list[StockRecommendation],
    reason: DisagreementReasonCode,
) -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion(claude_items), opinion(codex_items)
    )

    assert result.recommendations == []
    assert result.disagreements[0].reason_code is reason


def test_ranked_orders_buy_hold_sell_then_confidence() -> None:
    entries = [
        item("000001", Action.SELL, 0.9),
        item("000002", Action.BUY, 0.6),
        item("000003", Action.HOLD, 0.9, None, None),
        item("000004", Action.BUY, 0.8),
    ]
    result = RecommendationConsensusPolicy.reconcile(opinion(entries), opinion(entries))

    assert [entry.code for entry in result.ranked()] == [
        "000004",
        "000002",
        "000003",
        "000001",
    ]


def test_one_sided_recommendation_averages_both_available_scores() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion([item(score="1.5", expected="5")]),
        opinion(
            [
                item(
                    rating=Action.HOLD,
                    target=None,
                    stop=None,
                    score="0.5",
                    expected="-1",
                )
            ]
        ),
        ConsensusPolicyName.BUY_ONE_SIDE,
    )

    merged = result.recommendations[0]
    assert merged.consensus_basis is ConsensusBasis.CLAUDE_ONLY
    assert merged.score == Decimal("1.0")
    assert merged.expected_excess_return_10d == Decimal("2")


def test_disagreement_retains_model_scores_and_candidates_sort_stably() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion(
            [
                item("000002", score="1.5"),
                item("000001", score="1.0"),
            ]
        ),
        opinion(
            [
                item("000002", rating=Action.SELL, score="0.5"),
                item("000001", rating=Action.SELL, score="1.0"),
            ]
        ),
    )

    assert result.recommendations == []
    assert result.candidates[0].code == "000001"  # 동점이면 종목 코드 순
    assert result.candidates[0].score == Decimal("1.0")
    assert result.candidates[1].claude_score == Decimal("1.5")
    assert result.candidates[1].codex_score == Decimal("0.5")
    assert [item.code for item in result.top_candidates(5)] == ["000001", "000002"]


def test_candidates_average_present_values_and_include_single_model_scores() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion(
            [
                item("000001", score="0.5"),
                item("000002", score="2"),
            ]
        ),
        opinion(
            [
                item("000001", rating=Action.HOLD, target=None, stop=None, score="1"),
                item("000003", score="1"),
            ]
        ),
    )

    assert [(item.code, item.score) for item in result.top_candidates(5)] == [
        ("000002", Decimal("2")),
        ("000003", Decimal("1")),
        ("000001", Decimal("0.75")),
    ]
    assert result.candidates[0].claude_score == Decimal("2")
    assert result.candidates[0].codex_score is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"code": "ABC"},
        {"rating": "strong_buy"},
        {"confidence": 1.5},
        {"score": Decimal("2.1")},
        {"expected_excess_return_10d": Decimal("-50.1")},
        {"drivers": ["bad_driver"]},
        {"drivers": ["trend_up", "trend_up"]},
        {"key_risk_type": "bad_risk", "key_risk_note": "note"},
        {"key_risk_type": "event_risk"},
    ],
)
def test_domain_rejects_invalid_recommendation(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "code": "005930",
        "rating": "buy",
        "confidence": 0.5,
        "target_price": None,
        "invalidation_price": None,
        "reason": "r",
    }
    values.update(overrides)
    with pytest.raises(DomainError):
        StockRecommendation(**values)  # type: ignore[arg-type]


def response(**overrides: object) -> str:
    entry: dict[str, object] = {
        "code": "005930",
        "rating": "buy",
        "confidence": 0.75,
        "target_price": 120000,
        "invalidation_price": 95000.5,
        "reason": "trend",
        "score": 1.25,
        "expected_excess_return_10d": 4.5,
        "drivers": ["trend_up", "relative_strength"],
        "key_risk": {"type": "event_risk", "note": "실적 발표"},
    }
    entry.update(overrides)
    return json.dumps(
        {"summary": "s", "recommendations": [entry], "missing_info": ["x"]}
    )


def test_parser_converts_nullable_prices() -> None:
    parsed = parse_model_recommendation(
        response(rating="hold", target_price=None, invalidation_price=None), "codex"
    )

    assert parsed.model_id == "codex"
    assert parsed.recommendations[0].rating is Action.HOLD
    assert parsed.recommendations[0].target_price is None
    assert parsed.recommendations[0].score == Decimal("1.25")


def test_parser_keeps_decimal_prices() -> None:
    parsed = parse_model_recommendation(response(), "claude")

    assert parsed.recommendations[0].invalidation_price == Decimal("95000.5")
    assert parsed.recommendations[0].expected_excess_return_10d == Decimal("4.5")


def test_parser_accepts_optional_stock_name_and_ignores_it() -> None:
    parsed = parse_model_recommendation(response(name="삼성전자"), "claude")

    assert parsed.recommendations[0].code == "005930"


def test_parser_rejects_non_string_optional_stock_name() -> None:
    with pytest.raises(ValueError, match="name는 문자열"):
        parse_model_recommendation(response(name=123), "claude")


@pytest.mark.parametrize(
    "overrides",
    [
        {"score": 2.01},
        {"score": -2.1},
        {"expected_excess_return_10d": 50.1},
        {"drivers": ["not_a_driver"]},
        {"drivers": ["trend_up", "trend_up"]},
        {"drivers": ["trend_up", "trend_break", "overheated", "breakout"]},
        {"key_risk": {"type": "unknown", "note": "risk"}},
        {"key_risk": {"type": "event_risk"}},
        {"key_risk": "event_risk"},
    ],
)
def test_parser_fails_closed_for_invalid_relative_fields(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        parse_model_recommendation(response(**overrides), "claude")


@pytest.mark.parametrize(
    "overrides",
    [{"target_price": "120"}, {"confidence": True}, {"extra": 1}],
)
def test_parser_rejects_wrong_types(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_model_recommendation(response(**overrides), "claude")


def test_schema_requires_every_property_and_forbids_extra() -> None:
    item_schema = RECOMMENDATION_RESPONSE_SCHEMA["properties"]["recommendations"][
        "items"
    ]
    assert item_schema["properties"]["name"] == {"type": "string"}
    for schema in (
        RECOMMENDATION_RESPONSE_SCHEMA,
        item_schema,
    ):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])

    drivers_schema = item_schema["properties"]["drivers"]
    assert "uniqueItems" not in drivers_schema
    assert "maxItems" not in drivers_schema
    key_risk_schema = item_schema["properties"]["key_risk"]
    assert key_risk_schema["additionalProperties"] is False
    assert set(key_risk_schema["required"]) == set(key_risk_schema["properties"])


@pytest.mark.parametrize(
    "drivers",
    [
        ["trend_up", "trend_up"],
        ["trend_up", "trend_break", "overheated", "breakout"],
    ],
)
def test_parser_rejects_duplicate_or_excess_drivers_without_schema_limits(
    drivers: list[str],
) -> None:
    with pytest.raises(ValueError):
        parse_model_recommendation(response(drivers=drivers), "claude")


@pytest.mark.parametrize("policy", list(ConsensusPolicyName))
@pytest.mark.parametrize("aggregation", list(ConfidenceAggregation))
@pytest.mark.parametrize(
    ("claude_ratings", "codex_ratings", "strict_basis", "buy_basis"),
    [
        ((Action.BUY,), (Action.BUY,), ConsensusBasis.AGREED, ConsensusBasis.AGREED),
        ((Action.BUY,), (Action.HOLD,), None, ConsensusBasis.CLAUDE_ONLY),
        ((Action.BUY,), (), None, ConsensusBasis.CLAUDE_ONLY),
        ((Action.HOLD,), (Action.BUY,), None, ConsensusBasis.CODEX_ONLY),
        ((), (Action.BUY,), None, ConsensusBasis.CODEX_ONLY),
        ((Action.BUY,), (Action.SELL,), None, None),
        ((Action.SELL,), (Action.HOLD,), None, None),
        ((Action.SELL,), (Action.SELL,), ConsensusBasis.AGREED, ConsensusBasis.AGREED),
        (
            (Action.BUY, Action.BUY),
            (Action.BUY,),
            None,
            None,
        ),
    ],
)
def test_recommendation_policy_and_aggregation_matrix(
    policy: ConsensusPolicyName,
    aggregation: ConfidenceAggregation,
    claude_ratings: tuple[Action, ...],
    codex_ratings: tuple[Action, ...],
    strict_basis: ConsensusBasis | None,
    buy_basis: ConsensusBasis | None,
) -> None:
    def entries(ratings: tuple[Action, ...]) -> list[StockRecommendation]:
        return [
            item(
                rating=rating,
                target=None if rating is Action.HOLD else "120",
                stop=None if rating is Action.HOLD else "90",
            )
            for rating in ratings
        ]

    result = RecommendationConsensusPolicy.reconcile(
        opinion(entries(claude_ratings)),
        opinion(entries(codex_ratings)),
        policy,
        aggregation,
    )

    expected_basis = strict_basis if policy is ConsensusPolicyName.STRICT else buy_basis
    assert result.policy is policy
    assert result.confidence_aggregation is aggregation
    if expected_basis is None:
        assert result.recommendations == []
        assert len(result.disagreements) == 1
        expected_reason = (
            DisagreementReasonCode.DUPLICATE_PROPOSAL
            if len(claude_ratings) > 1 or len(codex_ratings) > 1
            else DisagreementReasonCode.ACTION_MISMATCH
        )
        assert result.disagreements[0].reason_code is expected_reason
    else:
        assert len(result.recommendations) == 1
        merged = result.recommendations[0]
        assert merged.consensus_basis is expected_basis
        assert merged.rating in (Action.BUY, Action.SELL)
        assert result.disagreements == []
        if expected_basis in (ConsensusBasis.CLAUDE_ONLY, ConsensusBasis.CODEX_ONLY):
            assert merged.confidence == 0.8
            assert merged.target_price == Decimal("120")
            if expected_basis is ConsensusBasis.CLAUDE_ONLY and codex_ratings:
                assert "[codex:hold]" in merged.reason
            if expected_basis is ConsensusBasis.CODEX_ONLY and claude_ratings:
                assert "[claude:hold]" in merged.reason


@pytest.mark.parametrize("policy", list(ConsensusPolicyName))
@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        (ConfidenceAggregation.MIN, 0.6),
        (ConfidenceAggregation.MEAN, 0.65),
    ],
)
def test_recommendation_confidence_aggregation_and_one_sided_value(
    policy: ConsensusPolicyName,
    aggregation: ConfidenceAggregation,
    expected: float,
) -> None:
    agreed = RecommendationConsensusPolicy.reconcile(
        opinion([item(confidence=0.6)]),
        opinion([item(confidence=0.7)]),
        policy,
        aggregation,
    )
    one_sided = RecommendationConsensusPolicy.reconcile(
        opinion([item(confidence=0.8)]),
        opinion([]),
        ConsensusPolicyName.BUY_ONE_SIDE,
        aggregation,
    )

    assert agreed.recommendations[0].confidence == expected
    assert one_sided.recommendations[0].confidence == 0.8
