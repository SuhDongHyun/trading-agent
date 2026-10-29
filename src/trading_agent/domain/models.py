"""매매 의사결정에 사용하는 도메인 모델."""

import math
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from trading_agent.domain.errors import DomainError


def _decimal(value: Decimal | int | float | str) -> Decimal:
    """금액과 지표를 일관된 Decimal 값으로 바꾼다."""

    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (TypeError, ValueError, ArithmeticError) as error:
        raise DomainError("숫자 값이 올바르지 않습니다.") from error
    if not result.is_finite():
        raise DomainError("숫자 값은 유한해야 합니다.")
    return result


@dataclass(frozen=True, slots=True)
class Position:
    """계좌가 보유한 종목과 평가 정보."""

    code: str
    name: str
    quantity: int
    average_price: Decimal | None = None
    current_price: Decimal | None = None
    market_value: Decimal | None = None
    profit_loss_rate: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise DomainError("보유 수량은 정수여야 합니다.")
        if self.quantity < 0:
            raise DomainError("보유 수량은 음수일 수 없습니다.")
        for field_name in (
            "average_price",
            "current_price",
            "market_value",
            "profit_loss_rate",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _decimal(value))

    @property
    def evaluation_amount(self) -> Decimal | None:
        """제공된 평가액 또는 현재가로 계산한 평가액을 반환한다."""

        if self.market_value is not None:
            return self.market_value
        if self.current_price is not None:
            return self.current_price * self.quantity
        return None


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """현금과 보유 종목으로 구성한 계좌 스냅샷."""

    cash: Decimal
    positions: list[Position]

    def __post_init__(self) -> None:
        cash = _decimal(self.cash)
        if cash < 0:
            raise DomainError("현금은 음수일 수 없습니다.")
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "positions", list(self.positions))

    @property
    def total_asset_value(self) -> Decimal | None:
        """모든 보유 종목을 평가할 수 있을 때 총자산을 계산한다."""

        total = self.cash
        for position in self.positions:
            value = position.evaluation_amount
            if value is None and position.quantity > 0:
                return None
            if value is not None:
                total += value
        return total


@dataclass(frozen=True, slots=True)
class StockSnapshot:
    """한 종목에 대해 요약한 시세, 지표, 뉴스."""

    code: str
    name: str
    current_price: Decimal | None = None
    change_percent: Decimal | None = None
    return_percent: Decimal | None = None
    moving_average_deviation: Decimal | None = None
    rsi: Decimal | None = None
    rsi_signal: str | None = None
    macd: Decimal | None = None
    macd_signal: Decimal | None = None
    volume_change_percent: Decimal | None = None
    news_headlines: list[str] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "current_price",
            "change_percent",
            "return_percent",
            "moving_average_deviation",
            "rsi",
            "macd",
            "macd_signal",
            "volume_change_percent",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _decimal(value))
        if self.news_headlines is not None:
            object.__setattr__(self, "news_headlines", list(self.news_headlines))


@dataclass(frozen=True, slots=True)
class MarketContext:
    """시장 지수와 거시 지표 요약."""

    index_summary: str | None = None
    volatility_index: Decimal | None = None
    fear_greed_index: Decimal | None = None
    exchange_rate: Decimal | None = None
    interest_rate: Decimal | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "volatility_index",
            "fear_greed_index",
            "exchange_rate",
            "interest_rate",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _decimal(value))


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """한 사이클의 계좌, 종목, 시장 데이터."""

    created_at: datetime
    account: AccountSnapshot
    stocks: list[StockSnapshot]
    market_context: MarketContext
    missing: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stocks", list(self.stocks))
        object.__setattr__(self, "missing", list(self.missing))


class Action(StrEnum):
    """모델이 제안할 수 있는 매매 동작."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """모델이 제시한 단일 매매 제안."""

    code: str
    action: Action
    quantity: int
    limit_price: Decimal
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str):
            raise DomainError("종목 코드는 문자열이어야 합니다.")
        if re.fullmatch(r"[0-9]{6}", self.code) is None:
            raise DomainError("종목 코드는 6자리 숫자여야 합니다.")
        try:
            action = Action(self.action)
        except (TypeError, ValueError) as error:
            raise DomainError("action은 buy, sell, hold 중 하나여야 합니다.") from error
        object.__setattr__(self, "action", action)
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise DomainError("매매 수량은 정수여야 합니다.")
        if action in (Action.BUY, Action.SELL) and self.quantity <= 0:
            raise DomainError("매매 수량은 0보다 커야 합니다.")
        if self.quantity < 0:
            raise DomainError("매매 수량은 음수일 수 없습니다.")
        if isinstance(self.limit_price, bool) or not isinstance(
            self.limit_price, (Decimal, int, float)
        ):
            raise DomainError("지정가는 숫자여야 합니다.")
        limit_price = _decimal(self.limit_price)
        if limit_price <= 0:
            raise DomainError("지정가는 0보다 커야 합니다.")
        object.__setattr__(self, "limit_price", limit_price)
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float, Decimal)
        ):
            raise DomainError("confidence는 숫자여야 합니다.")
        try:
            confidence = float(self.confidence)
        except (OverflowError, TypeError, ValueError) as error:
            raise DomainError("confidence 값이 올바르지 않습니다.") from error
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise DomainError("confidence는 0과 1 사이여야 합니다.")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class ModelDecision:
    """한 모델의 제안과 판단 요약."""

    model_id: str
    proposals: list[TradeProposal]
    summary: str
    missing_info: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposals", list(self.proposals))
        object.__setattr__(self, "missing_info", list(self.missing_info))
