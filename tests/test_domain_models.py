from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_agent.domain.errors import DomainError
from trading_agent.domain.models import (
    AccountSnapshot,
    Action,
    MarketContext,
    MarketSnapshot,
    ModelDecision,
    Position,
    StockSnapshot,
    TradeProposal,
)


def proposal(
    action: Action = Action.BUY,
    quantity: int = 2,
    limit_price: Decimal = Decimal("100"),
    confidence: float = 0.8,
    code: str = "005930",
) -> TradeProposal:
    return TradeProposal(
        code=code,
        action=action,
        quantity=quantity,
        limit_price=limit_price,
        confidence=confidence,
        reason="test",
    )


@pytest.mark.parametrize(
    ("action", "quantity"),
    [(Action.BUY, 0), (Action.BUY, -1), (Action.SELL, 0), (Action.SELL, -1)],
)
def test_trade_proposal_requires_positive_quantity_for_trades(
    action: Action, quantity: int
) -> None:
    with pytest.raises(DomainError):
        proposal(action=action, quantity=quantity)


@pytest.mark.parametrize("code", ["12345", "1234567", "ABCDEF", "１２３４５６"])
def test_trade_proposal_requires_six_ascii_digits(code: str) -> None:
    with pytest.raises(DomainError):
        proposal(code=code)


@pytest.mark.parametrize("code", [5930, None, []])
def test_trade_proposal_rejects_non_string_code(code: object) -> None:
    with pytest.raises(DomainError):
        proposal(code=code)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit_price", [0, -1, "NaN", "Infinity"])
def test_trade_proposal_requires_positive_finite_limit_price(
    limit_price: int | str,
) -> None:
    with pytest.raises(DomainError):
        proposal(limit_price=limit_price)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit_price", [True, "100", None, object()])
def test_trade_proposal_rejects_non_numeric_limit_price(limit_price: object) -> None:
    with pytest.raises(DomainError):
        proposal(limit_price=limit_price)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_trade_proposal_requires_confidence_between_zero_and_one(
    confidence: float,
) -> None:
    with pytest.raises(DomainError):
        proposal(confidence=confidence)


@pytest.mark.parametrize("confidence", [True, "0.9", None, object()])
def test_trade_proposal_rejects_non_numeric_confidence(confidence: object) -> None:
    with pytest.raises(DomainError):
        proposal(confidence=confidence)  # type: ignore[arg-type]


@pytest.mark.parametrize("quantity", [True, 1.5, "2"])
def test_trade_proposal_rejects_non_integer_quantity(quantity: object) -> None:
    with pytest.raises(DomainError):
        proposal(quantity=quantity)  # type: ignore[arg-type]


def test_trade_proposal_converts_supported_action_values() -> None:
    buy = proposal(action="buy")  # type: ignore[arg-type]
    hold = proposal(action="hold", quantity=0)  # type: ignore[arg-type]

    assert buy.action is Action.BUY
    assert hold.action is Action.HOLD


@pytest.mark.parametrize("action", ["BUY", "nonsense", None])
def test_trade_proposal_rejects_unsupported_action(action: object) -> None:
    with pytest.raises(DomainError):
        proposal(action=action)  # type: ignore[arg-type]


def test_hold_may_have_zero_quantity() -> None:
    assert proposal(action=Action.HOLD, quantity=0).quantity == 0


def test_account_total_asset_value_uses_provided_and_calculated_values() -> None:
    account = AccountSnapshot(
        cash=Decimal("1000"),
        positions=[
            Position(code="005930", name="A", quantity=2, current_price=Decimal("50")),
            Position(code="000660", name="B", quantity=1, market_value=Decimal("30")),
        ],
    )

    assert account.total_asset_value == Decimal("1130")


def test_account_total_asset_value_is_unknown_if_a_position_cannot_be_valued() -> None:
    account = AccountSnapshot(
        cash=Decimal("1000"),
        positions=[Position(code="005930", name="A", quantity=2)],
    )

    assert account.total_asset_value is None


def test_snapshot_and_decision_models_accept_optional_market_data() -> None:
    snapshot = MarketSnapshot(
        created_at=datetime.now(UTC),
        account=AccountSnapshot(cash=Decimal("100"), positions=[]),
        stocks=[StockSnapshot(code="005930", name="A")],
        market_context=MarketContext(),
        missing=["news"],
    )
    decision = ModelDecision("claude", [proposal()], "summary", [])

    assert snapshot.stocks[0].rsi is None
    assert decision.proposals[0].code == "005930"
