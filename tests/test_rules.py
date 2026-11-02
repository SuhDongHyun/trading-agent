from decimal import Decimal

from trading_agent.domain.models import (
    AccountSnapshot,
    Position,
    RuleType,
)
from trading_agent.domain.risk import RiskPolicy
from trading_agent.domain.rules import generate_rule_proposals


def policy(**overrides: object) -> RiskPolicy:
    values: dict[str, object] = {
        "max_position_weight": Decimal("0.5"),
        "max_order_amount": None,
        "max_order_ratio": Decimal("1"),
        "min_confidence": 0.7,
        "allowed_codes": {"005930"},
        "max_proposals_per_cycle": 5,
        "stop_loss_percent": Decimal("10"),
        "take_profit_percent": Decimal("20"),
        "take_profit_sell_fraction": Decimal("0.5"),
    }
    values.update(overrides)
    return RiskPolicy(**values)  # type: ignore[arg-type]


def test_stop_loss_triggers_at_boundary_and_precedes_rebalance() -> None:
    account = AccountSnapshot(
        Decimal("500"),
        [
            Position(
                "005930",
                "삼성전자",
                7,
                average_price=Decimal("100"),
                current_price=Decimal("90"),
            )
        ],
    )

    proposals = generate_rule_proposals(
        account, policy(max_position_weight=Decimal("0.3"))
    )

    assert len(proposals) == 1
    assert proposals[0].rule_type is RuleType.STOP_LOSS
    assert proposals[0].quantity == 7
    assert proposals[0].limit_price == Decimal("90")
    assert proposals[0].confidence == 1.0
    assert "-10.00%" in proposals[0].reason


def test_take_profit_triggers_at_boundary_and_precedes_rebalance() -> None:
    account = AccountSnapshot(
        Decimal("500"),
        [
            Position(
                "005930",
                "삼성전자",
                5,
                average_price=Decimal("100"),
                current_price=Decimal("120"),
            )
        ],
    )

    proposals = generate_rule_proposals(
        account, policy(max_position_weight=Decimal("0.4"))
    )

    assert len(proposals) == 1
    assert proposals[0].rule_type is RuleType.TAKE_PROFIT
    assert proposals[0].quantity == 2
    assert "20.00%" in proposals[0].reason


def test_take_profit_rounds_down_but_sells_at_least_one_share() -> None:
    account = AccountSnapshot(
        Decimal("1000"),
        [
            Position(
                "005930",
                "삼성전자",
                1,
                average_price=Decimal("100"),
                current_price=Decimal("120"),
            )
        ],
    )

    proposal = generate_rule_proposals(
        account, policy(max_position_weight=Decimal("1"))
    )[0]

    assert proposal.rule_type is RuleType.TAKE_PROFIT
    assert proposal.quantity == 1


def test_rebalance_uses_ceiling_to_reach_weight_limit() -> None:
    account = AccountSnapshot(
        Decimal("500"),
        [Position("005930", "삼성전자", 50, current_price=Decimal("20"))],
    )

    proposal = generate_rule_proposals(
        account,
        policy(
            stop_loss_percent=None,
            take_profit_percent=None,
            max_position_weight=Decimal("0.4"),
        ),
    )[0]

    assert proposal.rule_type is RuleType.REBALANCE
    assert proposal.quantity == 20
    assert "66.67%" in proposal.reason


def test_disabled_thresholds_and_missing_prices_produce_no_proposals() -> None:
    account = AccountSnapshot(
        Decimal("1000"),
        [Position("005930", "삼성전자", 1, average_price=Decimal("100"))],
    )

    assert (
        generate_rule_proposals(
            account,
            policy(
                stop_loss_percent=None,
                take_profit_percent=None,
                max_position_weight=Decimal("1"),
            ),
        )
        == []
    )
