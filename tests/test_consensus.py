from decimal import Decimal

import pytest

from trading_agent.domain.consensus import (
    ConsensusPolicy,
    DisagreementReasonCode,
)
from trading_agent.domain.models import Action, ModelDecision, TradeProposal


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


def test_one_sided_proposal_is_recorded() -> None:
    result = ConsensusPolicy.reconcile(decision([proposal(Action.BUY)]), decision([]))

    assert result.proposals == []
    assert result.disagreements[0].reason_code is DisagreementReasonCode.ONE_SIDED
    assert result.disagreements[0].claude_actions == (Action.BUY,)
    assert result.disagreements[0].codex_actions == ()


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
