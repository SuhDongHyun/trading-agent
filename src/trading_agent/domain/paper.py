"""가상 계좌의 체결과 평가 규칙."""

from dataclasses import dataclass, field, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import StrEnum

from trading_agent.domain.errors import DomainError
from trading_agent.domain.models import (
    AccountSnapshot,
    Action,
    Position,
    ProposalSource,
    TradeProposal,
    to_decimal,
)

_WON = Decimal(1)


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """체결 금액에 적용할 수수료·세금·슬리피지 비율."""

    commission_rate: Decimal = Decimal("0.00015")
    sell_tax_rate: Decimal = Decimal("0.002")
    slippage_rate: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for name in ("commission_rate", "sell_tax_rate", "slippage_rate"):
            value = to_decimal(getattr(self, name))
            if value < 0 or value >= 1:
                raise DomainError("수수료·세금·슬리피지 비율은 0 이상 1 미만입니다.")
            object.__setattr__(self, name, value)


class FillStatus(StrEnum):
    """가상 주문의 처리 결과."""

    FILLED = "filled"
    UNFILLED = "unfilled"


@dataclass(frozen=True, slots=True)
class PaperHolding:
    """가상 계좌가 보유한 종목과 평균 매입가(수수료 제외)."""

    code: str
    name: str
    quantity: int
    average_price: Decimal

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise DomainError("가상 보유 수량은 양수여야 합니다.")
        object.__setattr__(self, "average_price", to_decimal(self.average_price))


@dataclass(frozen=True, slots=True)
class PaperFill:
    """승인된 제안 하나를 가상 체결한 결과."""

    proposal: TradeProposal
    name: str
    status: FillStatus
    quote_price: Decimal | None
    quantity: int = 0
    fill_price: Decimal | None = None
    commission: Decimal = Decimal(0)
    tax: Decimal = Decimal(0)
    realized_pnl: Decimal | None = None
    note: str | None = None

    @property
    def gross_amount(self) -> Decimal:
        """수수료·세금을 빼기 전 체결 금액."""

        if self.fill_price is None:
            return Decimal(0)
        return self.fill_price * self.quantity


@dataclass(slots=True)
class PaperPortfolio:
    """초기 자금, 현금, 보유 종목으로 구성한 가상 계좌."""

    initial_cash: Decimal
    cash: Decimal
    holdings: dict[str, PaperHolding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.initial_cash = to_decimal(self.initial_cash)
        self.cash = to_decimal(self.cash)
        if self.initial_cash <= 0:
            raise DomainError("초기 자금은 0보다 커야 합니다.")
        if self.cash < 0:
            raise DomainError("가상 계좌 현금은 음수일 수 없습니다.")

    def account_snapshot(self) -> AccountSnapshot:
        """판단 사이클에 넣을 계좌 스냅샷을 만든다. 현재가는 스냅샷이 채운다."""

        return AccountSnapshot(
            cash=self.cash,
            positions=[
                Position(
                    code=holding.code,
                    name=holding.name,
                    quantity=holding.quantity,
                    average_price=holding.average_price,
                )
                for holding in self.holdings.values()
            ],
        )

    def market_value(self, prices: dict[str, Decimal]) -> Decimal:
        """주어진 가격으로 보유 종목 평가액을 계산한다. 가격이 없으면 평균가를 쓴다."""

        return sum(
            (
                prices.get(code, holding.average_price) * holding.quantity
                for code, holding in self.holdings.items()
            ),
            Decimal(0),
        )

    def execute(
        self,
        proposal: TradeProposal,
        name: str,
        quote_price: Decimal,
        fees: FeeSchedule,
    ) -> PaperFill:
        """현재가로 시장성 있는 지정가 주문만 체결하고 계좌를 갱신한다.

        매수는 현재가가 지정가 이하일 때, 합의 매도는 현재가가 지정가 이상일 때
        현재가(슬리피지 반영)로 체결한다. 손절·익절·리밸런싱 규칙 매도는 시장가로
        본다. 체결되지 않은 주문은 이번 사이클에서 취소된다.
        """

        quote_price = to_decimal(quote_price)
        if quote_price <= 0:
            raise DomainError("체결 기준 현재가는 0보다 커야 합니다.")
        if proposal.action is Action.BUY:
            return self._buy(proposal, name, quote_price, fees)
        if proposal.action is Action.SELL:
            return self._sell(proposal, name, quote_price, fees)
        return PaperFill(
            proposal, name, FillStatus.UNFILLED, quote_price, note="hold는 주문 없음"
        )

    def _buy(
        self,
        proposal: TradeProposal,
        name: str,
        quote_price: Decimal,
        fees: FeeSchedule,
    ) -> PaperFill:
        price = (quote_price * (1 + fees.slippage_rate)).quantize(
            _WON, rounding=ROUND_CEILING
        )
        if price > proposal.limit_price:
            return PaperFill(
                proposal,
                name,
                FillStatus.UNFILLED,
                quote_price,
                note="현재가가 매수 지정가보다 높아 미체결",
            )
        affordable = int(
            (self.cash / (price * (1 + fees.commission_rate))).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        quantity = min(proposal.quantity, affordable)
        if quantity <= 0:
            return PaperFill(
                proposal, name, FillStatus.UNFILLED, quote_price, note="현금 부족"
            )
        amount = price * quantity
        commission = _floor_won(amount * fees.commission_rate)
        self.cash -= amount + commission
        held = self.holdings.get(proposal.code)
        if held is None:
            self.holdings[proposal.code] = PaperHolding(
                proposal.code, name, quantity, price
            )
        else:
            total = held.quantity + quantity
            average = (held.average_price * held.quantity + amount) / total
            self.holdings[proposal.code] = replace(
                held, quantity=total, average_price=average
            )
        return PaperFill(
            proposal,
            name,
            FillStatus.FILLED,
            quote_price,
            quantity=quantity,
            fill_price=price,
            commission=commission,
            note=None if quantity == proposal.quantity else "현금 한도로 수량 축소",
        )

    def _sell(
        self,
        proposal: TradeProposal,
        name: str,
        quote_price: Decimal,
        fees: FeeSchedule,
    ) -> PaperFill:
        held = self.holdings.get(proposal.code)
        if held is None:
            return PaperFill(
                proposal, name, FillStatus.UNFILLED, quote_price, note="보유 수량 없음"
            )
        price = (quote_price * (1 - fees.slippage_rate)).quantize(
            _WON, rounding=ROUND_FLOOR
        )
        is_rule = proposal.source is ProposalSource.RULE
        if not is_rule and price < proposal.limit_price:
            return PaperFill(
                proposal,
                name,
                FillStatus.UNFILLED,
                quote_price,
                note="현재가가 매도 지정가보다 낮아 미체결",
            )
        quantity = min(proposal.quantity, held.quantity)
        amount = price * quantity
        commission = _floor_won(amount * fees.commission_rate)
        tax = _floor_won(amount * fees.sell_tax_rate)
        self.cash += amount - commission - tax
        realized = (price - held.average_price) * quantity - commission - tax
        remaining = held.quantity - quantity
        if remaining == 0:
            del self.holdings[proposal.code]
        else:
            self.holdings[proposal.code] = replace(held, quantity=remaining)
        return PaperFill(
            proposal,
            name,
            FillStatus.FILLED,
            quote_price,
            quantity=quantity,
            fill_price=price,
            commission=commission,
            tax=tax,
            realized_pnl=realized,
            note=None if quantity == proposal.quantity else "보유 수량으로 축소",
        )


def _floor_won(amount: Decimal) -> Decimal:
    """원 미만을 버린다."""

    return amount.quantize(_WON, rounding=ROUND_FLOOR)
