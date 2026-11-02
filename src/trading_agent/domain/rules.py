"""계좌 가격과 리스크 정책에서 규칙 기반 매매 제안을 만든다."""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from trading_agent.domain.models import (
    AccountSnapshot,
    Action,
    Position,
    ProposalSource,
    RuleType,
    TradeProposal,
)
from trading_agent.domain.risk import RiskPolicy


def generate_rule_proposals(
    account: AccountSnapshot, policy: RiskPolicy
) -> list[TradeProposal]:
    """손절, 익절, 비중 축소 순으로 종목별 규칙 제안을 만든다."""

    total_assets = account.total_asset_value
    proposals = []
    for position in account.positions:
        if position.quantity <= 0 or position.current_price is None:
            continue
        if position.current_price <= 0:
            continue

        return_percent = _return_percent(position)
        if (
            policy.stop_loss_percent is not None
            and return_percent is not None
            and return_percent <= -policy.stop_loss_percent
        ):
            proposals.append(
                _proposal(
                    position,
                    position.quantity,
                    RuleType.STOP_LOSS,
                    f"손절 규칙: 평균 매입가 대비 수익률 {return_percent:.2f}%가 "
                    f"기준 -{policy.stop_loss_percent}% 이하입니다.",
                )
            )
            continue

        if (
            policy.take_profit_percent is not None
            and return_percent is not None
            and return_percent >= policy.take_profit_percent
        ):
            quantity = max(
                1,
                int(
                    (
                        Decimal(position.quantity) * policy.take_profit_sell_fraction
                    ).to_integral_value(rounding=ROUND_FLOOR)
                ),
            )
            proposals.append(
                _proposal(
                    position,
                    quantity,
                    RuleType.TAKE_PROFIT,
                    f"익절 규칙: 평균 매입가 대비 수익률 {return_percent:.2f}%가 "
                    f"기준 {policy.take_profit_percent}% 이상이므로 "
                    f"보유량의 {policy.take_profit_sell_fraction * 100}%를 매도합니다.",
                )
            )
            continue

        if total_assets is None or total_assets <= 0:
            continue
        position_value = position.evaluation_amount
        if position_value is None:
            continue
        weight = position_value / total_assets
        if weight <= policy.max_position_weight:
            continue
        excess_value = position_value - total_assets * policy.max_position_weight
        quantity = min(
            position.quantity,
            int(
                (excess_value / position.current_price).to_integral_value(
                    rounding=ROUND_CEILING
                )
            ),
        )
        if quantity <= 0:
            continue
        proposals.append(
            _proposal(
                position,
                quantity,
                RuleType.REBALANCE,
                f"리밸런싱 규칙: 현재 비중 {weight * 100:.2f}%가 "
                f"한도 {policy.max_position_weight * 100}%를 초과해 "
                f"한도까지 {quantity}주를 매도합니다.",
            )
        )
    return proposals


def _return_percent(position: Position) -> Decimal | None:
    """계좌 손익률을 우선 사용하고 없으면 평균 매입가에서 계산한다."""

    if position.profit_loss_rate is not None:
        return position.profit_loss_rate
    if position.average_price is not None and position.average_price > 0:
        return (
            (position.current_price - position.average_price)
            / position.average_price
            * 100
        )
    return None


def _proposal(
    position: Position,
    quantity: int,
    rule_type: RuleType,
    reason: str,
) -> TradeProposal:
    """규칙 메타데이터를 붙인 전량 또는 일부 매도 제안을 만든다."""

    assert position.current_price is not None
    return TradeProposal(
        code=position.code,
        action=Action.SELL,
        quantity=quantity,
        limit_price=position.current_price,
        confidence=1.0,
        reason=reason,
        source=ProposalSource.RULE,
        rule_type=rule_type,
    )
