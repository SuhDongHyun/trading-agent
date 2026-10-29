"""제안의 계좌 및 정책 한도를 검토하는 순수 로직."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from trading_agent.domain.errors import DomainError
from trading_agent.domain.models import AccountSnapshot, Action, TradeProposal


class RiskReasonCode(StrEnum):
    """리스크 판정 사유 코드."""

    CODE_NOT_ALLOWED = "code_not_allowed"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    ORDER_AMOUNT_EXCEEDED = "order_amount_exceeded"
    POSITION_WEIGHT_EXCEEDED = "position_weight_exceeded"
    CASH_INSUFFICIENT = "cash_insufficient"
    POSITION_VALUE_UNAVAILABLE = "position_value_unavailable"
    ACCOUNT_VALUE_UNAVAILABLE = "account_value_unavailable"
    NOT_ENOUGH_SHARES = "not_enough_shares"
    POSITION_NOT_HELD = "position_not_held"
    DUPLICATE_CODE = "duplicate_code"
    CYCLE_LIMIT_EXCEEDED = "cycle_limit_exceeded"
    HOLD_SKIPPED = "hold_skipped"
    ACTION_NOT_SUPPORTED = "action_not_supported"


_REASON_MESSAGES = {
    RiskReasonCode.CODE_NOT_ALLOWED: "허용 목록에 없는 종목입니다.",
    RiskReasonCode.CONFIDENCE_TOO_LOW: "confidence가 최소 기준보다 낮습니다.",
    RiskReasonCode.ORDER_AMOUNT_EXCEEDED: "1회 주문 금액 한도를 초과합니다.",
    RiskReasonCode.POSITION_WEIGHT_EXCEEDED: "매수 후 종목 비중 한도를 초과합니다.",
    RiskReasonCode.CASH_INSUFFICIENT: "매수 금액이 현금 잔고를 초과합니다.",
    RiskReasonCode.POSITION_VALUE_UNAVAILABLE: (
        "보유 종목의 평가액을 계산할 수 없습니다."
    ),
    RiskReasonCode.ACCOUNT_VALUE_UNAVAILABLE: "총자산을 계산할 수 없습니다.",
    RiskReasonCode.NOT_ENOUGH_SHARES: "매도 수량이 보유 수량을 초과합니다.",
    RiskReasonCode.POSITION_NOT_HELD: "보유하지 않은 종목은 매도할 수 없습니다.",
    RiskReasonCode.DUPLICATE_CODE: "같은 종목의 첫 제안 이후 중복된 제안입니다.",
    RiskReasonCode.CYCLE_LIMIT_EXCEEDED: "사이클당 승인 가능한 제안 수를 초과합니다.",
    RiskReasonCode.HOLD_SKIPPED: "HOLD 제안은 리스크 검토 대상이 아닙니다.",
    RiskReasonCode.ACTION_NOT_SUPPORTED: "지원하지 않는 매매 동작입니다.",
}


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    """매매 제안에 적용할 리스크 한도."""

    max_position_weight: Decimal
    max_order_amount: Decimal
    min_confidence: float
    allowed_codes: set[str] | frozenset[str]
    max_proposals_per_cycle: int

    def __post_init__(self) -> None:
        for field_name in ("max_position_weight", "max_order_amount"):
            value = getattr(self, field_name)
            try:
                amount = value if isinstance(value, Decimal) else Decimal(str(value))
            except (TypeError, ValueError, ArithmeticError) as error:
                raise DomainError("리스크 금액 한도가 올바르지 않습니다.") from error
            if not amount.is_finite() or amount <= 0:
                raise DomainError("리스크 금액 한도는 0보다 커야 합니다.")
            object.__setattr__(self, field_name, amount)
        if self.max_position_weight > 1:
            raise DomainError("종목 비중 한도는 1 이하여야 합니다.")
        if not 0 <= self.min_confidence <= 1:
            raise DomainError("최소 confidence는 0과 1 사이여야 합니다.")
        if self.max_proposals_per_cycle <= 0:
            raise DomainError("사이클당 제안 수는 0보다 커야 합니다.")
        object.__setattr__(self, "allowed_codes", frozenset(self.allowed_codes))


@dataclass(frozen=True, slots=True)
class RiskIssue:
    """한 리스크 사유의 코드와 표시 문구."""

    code: RiskReasonCode
    message: str


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """제안의 승인, 거절 또는 제외 판정."""

    proposal: TradeProposal
    approved: bool
    reasons: list[RiskIssue]
    skipped: bool = False


class RiskGuard:
    """제안 순서에 따라 계좌 잔액과 정책 한도를 적용해 검토한다."""

    @staticmethod
    def review(
        proposals: list[TradeProposal],
        account: AccountSnapshot,
        policy: RiskPolicy,
    ) -> list[RiskVerdict]:
        """제안 순서를 보존해 리스크 판정을 반환한다."""

        verdicts: list[RiskVerdict] = []
        seen_codes: set[str] = set()
        approved_count = 0
        remaining_cash = account.cash

        for proposal in proposals:
            if proposal.action is Action.HOLD:
                verdicts.append(
                    RiskVerdict(
                        proposal=proposal,
                        approved=False,
                        reasons=[_issue(RiskReasonCode.HOLD_SKIPPED)],
                        skipped=True,
                    )
                )
                continue

            reasons: list[RiskIssue] = []
            if proposal.code in seen_codes:
                reasons.append(_issue(RiskReasonCode.DUPLICATE_CODE))
            else:
                seen_codes.add(proposal.code)

            if proposal.code not in policy.allowed_codes:
                reasons.append(_issue(RiskReasonCode.CODE_NOT_ALLOWED))
            if proposal.confidence < policy.min_confidence:
                reasons.append(_issue(RiskReasonCode.CONFIDENCE_TOO_LOW))

            order_amount = proposal.limit_price * proposal.quantity
            if order_amount > policy.max_order_amount:
                reasons.append(_issue(RiskReasonCode.ORDER_AMOUNT_EXCEEDED))

            if proposal.action is Action.BUY:
                if order_amount > remaining_cash:
                    reasons.append(_issue(RiskReasonCode.CASH_INSUFFICIENT))
                _review_position_weight(proposal, account, policy, reasons)
            elif proposal.action is Action.SELL:
                held_quantity = sum(
                    position.quantity
                    for position in account.positions
                    if position.code == proposal.code
                )
                if held_quantity == 0:
                    reasons.append(_issue(RiskReasonCode.POSITION_NOT_HELD))
                elif proposal.quantity > held_quantity:
                    reasons.append(_issue(RiskReasonCode.NOT_ENOUGH_SHARES))
            else:
                reasons.append(_issue(RiskReasonCode.ACTION_NOT_SUPPORTED))

            if approved_count >= policy.max_proposals_per_cycle:
                reasons.append(_issue(RiskReasonCode.CYCLE_LIMIT_EXCEEDED))

            approved = not reasons
            if approved:
                approved_count += 1
                if proposal.action is Action.BUY:
                    remaining_cash -= order_amount
            verdicts.append(
                RiskVerdict(proposal=proposal, approved=approved, reasons=reasons)
            )

        return verdicts


def _review_position_weight(
    proposal: TradeProposal,
    account: AccountSnapshot,
    policy: RiskPolicy,
    reasons: list[RiskIssue],
) -> None:
    """매수 후 비중을 계산할 수 있을 때 정책 한도를 확인한다."""

    current_positions = [
        position for position in account.positions if position.code == proposal.code
    ]
    current_value = Decimal(0)
    for position in current_positions:
        value = position.evaluation_amount
        if value is None and position.quantity > 0:
            reasons.append(_issue(RiskReasonCode.POSITION_VALUE_UNAVAILABLE))
            return
        if value is not None:
            current_value += value

    total_assets = account.total_asset_value
    if total_assets is None or total_assets <= 0:
        reasons.append(_issue(RiskReasonCode.ACCOUNT_VALUE_UNAVAILABLE))
        return

    projected_weight = (
        current_value + proposal.limit_price * proposal.quantity
    ) / total_assets
    if projected_weight > policy.max_position_weight:
        reasons.append(_issue(RiskReasonCode.POSITION_WEIGHT_EXCEEDED))


def _issue(code: RiskReasonCode) -> RiskIssue:
    """사유 코드에서 사람이 읽을 메시지를 만든다."""

    return RiskIssue(code=code, message=_REASON_MESSAGES[code])
