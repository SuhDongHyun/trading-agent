from decimal import Decimal

import pytest

from trading_agent.domain.consensus import (
    ConsensusPolicy,
    DisagreementReasonCode,
)
from trading_agent.domain.models import (
    Action,
    ConfidenceAggregation,
    ConsensusBasis,
    ConsensusPolicyName,
    ModelDecision,
    TradeProposal,
)


def proposal(
    action: Action,
    *,
    code: str = "005930",
    quantity: int = 5,
    price: str = "100",
    confidence: float = 0.9,
    reason: str = "signal",
) -> TradeProposal:
    return TradeProposal(
        code=code,
        action=action,
        quantity=quantity,
        limit_price=Decimal(price),
        confidence=confidence,
        reason=reason,
    )


def decision(
    proposals: list[TradeProposal],
    *,
    summary: str = "summary",
    missing: list[str] | None = None,
) -> ModelDecision:
    return ModelDecision("test", proposals, summary, missing or [])


@pytest.mark.parametrize(
    ("action", "claude_price", "codex_price", "expected_price"),
    [
        (Action.BUY, "101", "99", Decimal("99")),
        (Action.SELL, "101", "99", Decimal("101")),
    ],
)
def test_matching_trade_actions_merge_conservatively(
    action: Action,
    claude_price: str,
    codex_price: str,
    expected_price: Decimal,
) -> None:
    claude = decision(
        [
            proposal(
                action,
                quantity=4,
                price=claude_price,
                confidence=0.8,
                reason="claude signal",
            )
        ],
        summary="claude summary",
        missing=["news", "market:vix"],
    )
    codex = decision(
        [
            proposal(
                action,
                quantity=2,
                price=codex_price,
                confidence=0.7,
                reason="codex signal",
            )
        ],
        summary="codex summary",
        missing=["market:vix", "daily-bars"],
    )

    result = ConsensusPolicy.reconcile(claude, codex)

    assert len(result.proposals) == 1
    merged = result.proposals[0]
    assert merged.quantity == 2
    assert merged.limit_price == expected_price
    assert merged.confidence == 0.7
    assert merged.reason == "[claude] claude signal / [codex] codex signal"
    assert result.disagreements == []
    assert result.missing_info == ["news", "market:vix", "daily-bars"]
    assert result.claude_summary == "claude summary"
    assert result.codex_summary == "codex summary"


def test_trade_against_omitted_stock_is_action_mismatch() -> None:
    result = ConsensusPolicy.reconcile(decision([proposal(Action.BUY)]), decision([]))

    assert result.proposals == []
    assert result.disagreements[0].reason_code is DisagreementReasonCode.ACTION_MISMATCH
    assert result.disagreements[0].claude_actions == (Action.BUY,)
    assert result.disagreements[0].codex_actions == ()


@pytest.mark.parametrize("hold_side", ["claude", "codex"])
def test_hold_and_omitted_stock_are_the_same_opinion(hold_side: str) -> None:
    hold = decision([proposal(Action.HOLD, quantity=0)])
    omitted = decision([])
    claude, codex = (hold, omitted) if hold_side == "claude" else (omitted, hold)

    result = ConsensusPolicy.reconcile(claude, codex)

    assert result.proposals == []
    assert result.disagreements == []


@pytest.mark.parametrize(
    ("claude_action", "codex_action"),
    [(Action.BUY, Action.SELL), (Action.BUY, Action.HOLD)],
)
def test_different_actions_are_not_adopted(
    claude_action: Action, codex_action: Action
) -> None:
    result = ConsensusPolicy.reconcile(
        decision([proposal(claude_action)]), decision([proposal(codex_action)])
    )

    assert result.proposals == []
    assert result.disagreements[0].reason_code is DisagreementReasonCode.ACTION_MISMATCH


def test_both_hold_and_both_missing_need_no_disagreement_record() -> None:
    holds = ConsensusPolicy.reconcile(
        decision([proposal(Action.HOLD, quantity=0)]),
        decision([proposal(Action.HOLD, quantity=0)]),
    )
    missing = ConsensusPolicy.reconcile(decision([]), decision([]))

    assert holds.proposals == []
    assert holds.disagreements == []
    assert missing.disagreements == []


def test_duplicate_proposal_in_either_model_is_disagreement() -> None:
    result = ConsensusPolicy.reconcile(
        decision([proposal(Action.BUY), proposal(Action.BUY, quantity=3)]),
        decision([proposal(Action.BUY)]),
    )

    assert result.proposals == []
    assert result.disagreements[0].reason_code is (
        DisagreementReasonCode.DUPLICATE_PROPOSAL
    )
    assert result.disagreements[0].claude_actions == (Action.BUY, Action.BUY)
    assert result.disagreements[0].codex_actions == (Action.BUY,)


def test_duplicate_in_second_model_is_also_rejected() -> None:
    result = ConsensusPolicy.reconcile(
        decision([proposal(Action.SELL)]),
        decision([proposal(Action.SELL), proposal(Action.SELL, quantity=2)]),
    )

    assert result.proposals == []
    assert result.disagreements[0].reason_code is (
        DisagreementReasonCode.DUPLICATE_PROPOSAL
    )


def test_disagreements_preserve_first_seen_stock_order() -> None:
    result = ConsensusPolicy.reconcile(
        decision(
            [
                proposal(Action.BUY, code="005930"),
                proposal(Action.SELL, code="000660"),
            ]
        ),
        decision([proposal(Action.SELL, code="035420")]),
    )

    assert [item.code for item in result.disagreements] == [
        "005930",
        "000660",
        "035420",
    ]


@pytest.mark.parametrize("policy", list(ConsensusPolicyName))
@pytest.mark.parametrize("aggregation", list(ConfidenceAggregation))
@pytest.mark.parametrize(
    ("claude_actions", "codex_actions", "strict_basis", "buy_basis"),
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
def test_policy_and_aggregation_matrix(
    policy: ConsensusPolicyName,
    aggregation: ConfidenceAggregation,
    claude_actions: tuple[Action, ...],
    codex_actions: tuple[Action, ...],
    strict_basis: ConsensusBasis | None,
    buy_basis: ConsensusBasis | None,
) -> None:
    def entries(actions: tuple[Action, ...]) -> list[TradeProposal]:
        return [
            proposal(action, quantity=0 if action is Action.HOLD else 5)
            for action in actions
        ]

    result = ConsensusPolicy.reconcile(
        decision(entries(claude_actions)),
        decision(entries(codex_actions)),
        policy,
        aggregation,
    )

    expected_basis = strict_basis if policy is ConsensusPolicyName.STRICT else buy_basis
    assert result.policy is policy
    assert result.confidence_aggregation is aggregation
    if expected_basis is None:
        assert result.proposals == []
        assert len(result.disagreements) == 1
        expected_reason = (
            DisagreementReasonCode.DUPLICATE_PROPOSAL
            if len(claude_actions) > 1 or len(codex_actions) > 1
            else DisagreementReasonCode.ACTION_MISMATCH
        )
        assert result.disagreements[0].reason_code is expected_reason
    else:
        assert len(result.proposals) == 1
        assert result.proposals[0].consensus_basis is expected_basis
        assert result.disagreements == []
        if expected_basis in (ConsensusBasis.CLAUDE_ONLY, ConsensusBasis.CODEX_ONLY):
            assert result.proposals[0].quantity == 5
            if expected_basis is ConsensusBasis.CLAUDE_ONLY and codex_actions:
                assert "[codex:hold]" in result.proposals[0].reason
            if expected_basis is ConsensusBasis.CODEX_ONLY and claude_actions:
                assert "[claude:hold]" in result.proposals[0].reason


@pytest.mark.parametrize("policy", list(ConsensusPolicyName))
@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        (ConfidenceAggregation.MIN, 0.6),
        (ConfidenceAggregation.MEAN, 0.65),
    ],
)
def test_confidence_aggregation_applies_only_to_agreed_items(
    policy: ConsensusPolicyName,
    aggregation: ConfidenceAggregation,
    expected: float,
) -> None:
    agreed = ConsensusPolicy.reconcile(
        decision([proposal(Action.BUY, confidence=0.6)]),
        decision([proposal(Action.BUY, confidence=0.7)]),
        policy,
        aggregation,
    )
    one_sided = ConsensusPolicy.reconcile(
        decision([proposal(Action.BUY, confidence=0.8)]),
        decision([]),
        ConsensusPolicyName.BUY_ONE_SIDE,
        aggregation,
    )

    assert agreed.proposals[0].confidence == expected
    assert one_sided.proposals[0].confidence == 0.8
