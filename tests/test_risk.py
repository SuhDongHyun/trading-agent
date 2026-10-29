from decimal import Decimal

import pytest

from trading_agent.domain.models import AccountSnapshot, Action, Position, TradeProposal
from trading_agent.domain.risk import (
    RiskGuard,
    RiskPolicy,
    RiskReasonCode,
)


def proposal(
    code: str = "005930",
    action: Action = Action.BUY,
    quantity: int = 2,
    price: str = "100",
    confidence: float = 0.8,
) -> TradeProposal:
    return TradeProposal(
        code=code,
        action=action,
        quantity=quantity,
        limit_price=Decimal(price),
        confidence=confidence,
        reason="test",
    )


def policy(**overrides: object) -> RiskPolicy:
    values: dict[str, object] = {
        "max_position_weight": Decimal("0.5"),
        "max_order_amount": Decimal("1000"),
        "min_confidence": 0.7,
        "allowed_codes": {"005930", "000660"},
        "max_proposals_per_cycle": 5,
    }
    values.update(overrides)
    return RiskPolicy(**values)  # type: ignore[arg-type]


def account(
    cash: str = "1000",
    positions: list[Position] | None = None,
) -> AccountSnapshot:
    return AccountSnapshot(cash=Decimal(cash), positions=positions or [])


def reason_codes(verdict: object) -> set[RiskReasonCode]:
    return {issue.code for issue in verdict.reasons}  # type: ignore[attr-defined]


def test_allowed_code_and_confidence_boundary_are_approved() -> None:
    verdict = RiskGuard.review(
        [proposal(confidence=0.7)],
        account(),
        policy(max_position_weight=Decimal("0.2")),
    )[0]

    assert verdict.approved
    assert verdict.reasons == []


def test_disallowed_code_is_rejected() -> None:
    verdict = RiskGuard.review([proposal(code="123456")], account(), policy())[0]

    assert RiskReasonCode.CODE_NOT_ALLOWED in reason_codes(verdict)


def test_low_confidence_is_rejected_but_equal_threshold_passes() -> None:
    verdicts = RiskGuard.review(
        [proposal(confidence=0.69), proposal(code="000660", confidence=0.7)],
        account(),
        policy(),
    )

    assert RiskReasonCode.CONFIDENCE_TOO_LOW in reason_codes(verdicts[0])
    assert verdicts[1].approved


def test_order_amount_equal_to_limit_passes_and_excess_is_rejected() -> None:
    verdicts = RiskGuard.review(
        [proposal(quantity=10, price="100"), proposal(code="000660", quantity=11)],
        account(),
        policy(
            max_order_amount=Decimal("1000"),
            max_position_weight=Decimal("1"),
        ),
    )

    assert verdicts[0].approved
    assert RiskReasonCode.ORDER_AMOUNT_EXCEEDED in reason_codes(verdicts[1])


def test_buy_at_position_weight_boundary_passes_and_excess_is_rejected() -> None:
    verdicts = RiskGuard.review(
        [proposal(quantity=5, price="100"), proposal(code="000660", quantity=6)],
        account(cash="1000"),
        policy(max_position_weight=Decimal("0.5")),
    )

    assert verdicts[0].approved
    assert RiskReasonCode.POSITION_WEIGHT_EXCEEDED in reason_codes(verdicts[1])


def test_buy_cash_boundary_passes_and_excess_is_rejected() -> None:
    verdicts = RiskGuard.review(
        [proposal(quantity=10, price="100"), proposal(code="000660", quantity=11)],
        account(cash="1000"),
        policy(max_position_weight=Decimal("1")),
    )

    assert verdicts[0].approved
    assert RiskReasonCode.CASH_INSUFFICIENT in reason_codes(verdicts[1])


def test_cycle_buy_cash_total_equal_to_balance_is_approved() -> None:
    verdicts = RiskGuard.review(
        [
            proposal(quantity=6, price="100"),
            proposal(code="000660", quantity=4, price="100"),
        ],
        account(cash="1000"),
        policy(max_position_weight=Decimal("1")),
    )

    assert all(verdict.approved for verdict in verdicts)


def test_cycle_buy_exceeding_balance_rejects_later_buy() -> None:
    verdicts = RiskGuard.review(
        [
            proposal(quantity=6, price="100"),
            proposal(code="000660", quantity=6, price="100"),
        ],
        account(cash="1000"),
        policy(max_position_weight=Decimal("1")),
    )

    assert verdicts[0].approved
    assert not verdicts[1].approved
    assert RiskReasonCode.CASH_INSUFFICIENT in reason_codes(verdicts[1])


def test_risk_guard_rejects_unknown_action_fail_closed() -> None:
    invalid_proposal = proposal()
    object.__setattr__(invalid_proposal, "action", "nonsense")

    verdict = RiskGuard.review([invalid_proposal], account(), policy())[0]

    assert not verdict.approved
    assert RiskReasonCode.ACTION_NOT_SUPPORTED in reason_codes(verdict)


def test_buy_rejects_when_existing_position_value_is_unknown() -> None:
    verdict = RiskGuard.review(
        [proposal()],
        account(positions=[Position("005930", "A", 1)]),
        policy(),
    )[0]

    assert RiskReasonCode.POSITION_VALUE_UNAVAILABLE in reason_codes(verdict)


def test_sell_requires_a_holding_and_cannot_exceed_held_quantity() -> None:
    verdicts = RiskGuard.review(
        [
            proposal(action=Action.SELL),
            proposal(code="000660", action=Action.SELL),
            proposal(code="000660", action=Action.SELL, quantity=3),
        ],
        account(positions=[Position("000660", "B", 2, current_price=Decimal("100"))]),
        policy(),
    )

    assert RiskReasonCode.POSITION_NOT_HELD in reason_codes(verdicts[0])
    assert verdicts[1].approved
    assert RiskReasonCode.NOT_ENOUGH_SHARES in reason_codes(verdicts[2])


def test_duplicate_code_after_first_actionable_proposal_is_rejected() -> None:
    verdicts = RiskGuard.review(
        [proposal(), proposal(code="005930", action=Action.SELL)],
        account(positions=[Position("005930", "A", 5, current_price=Decimal("100"))]),
        policy(),
    )

    assert verdicts[0].approved
    assert RiskReasonCode.DUPLICATE_CODE in reason_codes(verdicts[1])


def test_hold_is_skipped_without_using_cycle_capacity() -> None:
    verdicts = RiskGuard.review(
        [
            proposal(action=Action.HOLD, quantity=0),
            proposal(code="000660"),
        ],
        account(),
        policy(max_proposals_per_cycle=1),
    )

    assert verdicts[0].skipped
    assert RiskReasonCode.HOLD_SKIPPED in reason_codes(verdicts[0])
    assert verdicts[1].approved


def test_cycle_limit_applies_after_maximum_approved_proposals() -> None:
    verdicts = RiskGuard.review(
        [proposal(), proposal(code="000660")],
        account(),
        policy(max_proposals_per_cycle=1),
    )

    assert verdicts[0].approved
    assert RiskReasonCode.CYCLE_LIMIT_EXCEEDED in reason_codes(verdicts[1])


@pytest.mark.parametrize(
    "policy_values",
    [
        {"max_position_weight": Decimal("0")},
        {"max_position_weight": Decimal("1.1")},
        {"max_order_amount": Decimal("0")},
        {"min_confidence": 1.1},
        {"max_proposals_per_cycle": 0},
    ],
)
def test_risk_policy_rejects_invalid_limits(policy_values: dict[str, object]) -> None:
    from trading_agent.domain.errors import DomainError

    with pytest.raises(DomainError):
        policy(**policy_values)
