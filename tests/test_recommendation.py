import json
from decimal import Decimal

import pytest

from trading_agent.application.recommendation_schema import (
    RECOMMENDATION_RESPONSE_SCHEMA,
    parse_model_recommendation,
)
from trading_agent.domain.consensus import DisagreementReasonCode
from trading_agent.domain.errors import DomainError
from trading_agent.domain.models import Action
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
) -> StockRecommendation:
    return StockRecommendation(
        code=code,
        rating=rating,
        confidence=confidence,
        target_price=None if target is None else Decimal(target),
        stop_loss_price=None if stop is None else Decimal(stop),
        reason=f"{rating.value} reason",
    )


def opinion(items: list[StockRecommendation]) -> ModelRecommendation:
    return ModelRecommendation("fake", items, "summary", ["info"])


def test_matching_buy_keeps_lower_target_higher_stop_and_lower_confidence() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion([item(confidence=0.9, target="130", stop="88")]),
        opinion([item(confidence=0.7, target="120", stop="92")]),
    )

    merged = result.recommendations[0]
    assert merged.rating is Action.BUY
    assert merged.confidence == 0.7
    assert merged.target_price == Decimal("120")
    assert merged.stop_loss_price == Decimal("92")
    assert result.disagreements == []
    assert result.missing_info == ["info"]


def test_matching_sell_keeps_higher_target_and_lower_stop() -> None:
    result = RecommendationConsensusPolicy.reconcile(
        opinion([item(rating=Action.SELL, target="80", stop="115")]),
        opinion([item(rating=Action.SELL, target="85", stop="110")]),
    )

    merged = result.recommendations[0]
    assert merged.target_price == Decimal("85")
    assert merged.stop_loss_price == Decimal("110")


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


@pytest.mark.parametrize(
    "overrides",
    [{"code": "ABC"}, {"rating": "strong_buy"}, {"confidence": 1.5}],
)
def test_domain_rejects_invalid_recommendation(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "code": "005930",
        "rating": "buy",
        "confidence": 0.5,
        "target_price": None,
        "stop_loss_price": None,
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
        "stop_loss_price": 95000.5,
        "reason": "trend",
    }
    entry.update(overrides)
    return json.dumps(
        {"summary": "s", "recommendations": [entry], "missing_info": ["x"]}
    )


def test_parser_converts_nullable_prices() -> None:
    parsed = parse_model_recommendation(
        response(rating="hold", target_price=None, stop_loss_price=None), "codex"
    )

    assert parsed.model_id == "codex"
    assert parsed.recommendations[0].rating is Action.HOLD
    assert parsed.recommendations[0].target_price is None


def test_parser_keeps_decimal_prices() -> None:
    parsed = parse_model_recommendation(response(), "claude")

    assert parsed.recommendations[0].stop_loss_price == Decimal("95000.5")


@pytest.mark.parametrize(
    "overrides",
    [{"target_price": "120"}, {"confidence": True}, {"extra": 1}],
)
def test_parser_rejects_wrong_types(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_model_recommendation(response(**overrides), "claude")


def test_schema_requires_every_property_and_forbids_extra() -> None:
    for schema in (
        RECOMMENDATION_RESPONSE_SCHEMA,
        RECOMMENDATION_RESPONSE_SCHEMA["properties"]["recommendations"]["items"],
    ):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
