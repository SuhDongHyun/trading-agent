"""매매 의사결정에 사용하는 도메인 모델."""

import math
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from trading_agent.domain.errors import DomainError


def to_decimal(value: Decimal | int | float | str) -> Decimal:
    """금액과 지표를 일관된 Decimal 값으로 바꾼다."""

    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (TypeError, ValueError, ArithmeticError) as error:
        raise DomainError("숫자 값이 올바르지 않습니다.") from error
    if not result.is_finite():
        raise DomainError("숫자 값은 유한해야 합니다.")
    return result


def require_stock_code(code: object) -> str:
    """6자리 숫자 종목 코드를 확인한다."""

    if not isinstance(code, str):
        raise DomainError("종목 코드는 문자열이어야 합니다.")
    if re.fullmatch(r"[0-9]{6}", code) is None:
        raise DomainError("종목 코드는 6자리 숫자여야 합니다.")
    return code


def require_action(value: object) -> "Action":
    """buy, sell, hold 중 하나인 동작을 확인한다."""

    try:
        return Action(value)
    except (TypeError, ValueError) as error:
        raise DomainError("action은 buy, sell, hold 중 하나여야 합니다.") from error


def require_confidence(value: object) -> float:
    """0과 1 사이의 유한한 확신도를 확인한다."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise DomainError("confidence는 숫자여야 합니다.")
    try:
        confidence = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise DomainError("confidence 값이 올바르지 않습니다.") from error
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise DomainError("confidence는 0과 1 사이여야 합니다.")
    return confidence


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
                object.__setattr__(self, field_name, to_decimal(value))

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
        cash = to_decimal(self.cash)
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
    return_5d_percent: Decimal | None = None
    return_20d_percent: Decimal | None = None
    ma20_deviation_percent: Decimal | None = None
    ma60_deviation_percent: Decimal | None = None
    rsi: Decimal | None = None
    rsi_signal: str | None = None
    macd: Decimal | None = None
    macd_signal_value: Decimal | None = None
    macd_signal: str | None = None
    volume_change_percent: Decimal | None = None
    atr14_percent: Decimal | None = None
    realized_volatility_20d_percent: Decimal | None = None
    high_20d_distance_percent: Decimal | None = None
    low_20d_distance_percent: Decimal | None = None
    high_52w_position_percent: Decimal | None = None
    high_52w_period_trading_days: int | None = None
    news_headlines: list[str] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "current_price",
            "change_percent",
            "return_5d_percent",
            "return_20d_percent",
            "ma20_deviation_percent",
            "ma60_deviation_percent",
            "rsi",
            "macd",
            "macd_signal_value",
            "volume_change_percent",
            "atr14_percent",
            "realized_volatility_20d_percent",
            "high_20d_distance_percent",
            "low_20d_distance_percent",
            "high_52w_position_percent",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, to_decimal(value))
        if self.high_52w_period_trading_days is not None and (
            not isinstance(self.high_52w_period_trading_days, int)
            or isinstance(self.high_52w_period_trading_days, bool)
            or self.high_52w_period_trading_days <= 0
        ):
            raise DomainError("52주 위치 계산 거래일 수는 양의 정수여야 합니다.")
        if self.news_headlines is not None:
            object.__setattr__(self, "news_headlines", list(self.news_headlines))


@dataclass(frozen=True, slots=True)
class MarketContext:
    """시장 지수와 거시 지표 요약."""

    kospi_change_percent: Decimal | None = None
    kosdaq_change_percent: Decimal | None = None
    sp500_change_percent: Decimal | None = None
    volatility_index: Decimal | None = None
    vkospi: Decimal | None = None
    fear_greed_index: Decimal | None = None
    exchange_rate: Decimal | None = None
    korea_3y_yield: Decimal | None = None
    us_10y_yield: Decimal | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "volatility_index",
            "kospi_change_percent",
            "kosdaq_change_percent",
            "sp500_change_percent",
            "vkospi",
            "fear_greed_index",
            "exchange_rate",
            "korea_3y_yield",
            "us_10y_yield",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, to_decimal(value))


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """한 사이클의 계좌, 종목, 시장 데이터."""

    created_at: datetime
    account: AccountSnapshot
    stocks: list[StockSnapshot]
    market_context: MarketContext
    missing: list[str]
    as_of_trading_date: str | None = None
    market_session: str = "closed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stocks", list(self.stocks))
        object.__setattr__(self, "missing", list(self.missing))


@dataclass(frozen=True, slots=True)
class ResearchSnapshot:
    """계좌와 무관하게 종목을 평가하기 위한 종목·시장 데이터."""

    created_at: datetime
    stocks: list[StockSnapshot]
    market_context: MarketContext
    missing: list[str]
    as_of_trading_date: str | None = None
    market_session: str = "closed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stocks", list(self.stocks))
        object.__setattr__(self, "missing", list(self.missing))


class Action(StrEnum):
    """모델이 제안할 수 있는 매매 동작이자 종목 투자의견."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ConsensusPolicyName(StrEnum):
    """두 모델 의견을 합치는 정책 이름."""

    STRICT = "strict"
    BUY_ONE_SIDE = "buy_one_side"


class ConfidenceAggregation(StrEnum):
    """양쪽 동의 시 확신도를 집계하는 방법."""

    MIN = "min"
    MEAN = "mean"


class ConsensusBasis(StrEnum):
    """채택된 합의 항목의 근거 범위."""

    AGREED = "agreed"
    CLAUDE_ONLY = "claude_only"
    CODEX_ONLY = "codex_only"


class ProposalSource(StrEnum):
    """매매 제안이 만들어진 경로."""

    CONSENSUS = "consensus"
    RULE = "rule"


class RuleType(StrEnum):
    """규칙 기반 매매 제안의 종류."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    REBALANCE = "rebalance"


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """모델이 제시한 단일 매매 제안."""

    code: str
    action: Action
    quantity: int
    limit_price: Decimal
    confidence: float
    reason: str
    source: ProposalSource = ProposalSource.CONSENSUS
    rule_type: RuleType | None = None
    consensus_basis: ConsensusBasis | None = None

    def __post_init__(self) -> None:
        require_stock_code(self.code)
        action = require_action(self.action)
        object.__setattr__(self, "action", action)
        try:
            source = ProposalSource(self.source)
            rule_type = RuleType(self.rule_type) if self.rule_type is not None else None
        except (TypeError, ValueError) as error:
            raise DomainError(
                "제안 출처 또는 규칙 종류가 올바르지 않습니다."
            ) from error
        if (source is ProposalSource.RULE) != (rule_type is not None):
            raise DomainError("규칙 제안에는 규칙 종류가 필요합니다.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "rule_type", rule_type)
        if self.consensus_basis is not None:
            try:
                basis = ConsensusBasis(self.consensus_basis)
            except (TypeError, ValueError) as error:
                raise DomainError("합의 근거가 올바르지 않습니다.") from error
            object.__setattr__(self, "consensus_basis", basis)
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
        limit_price = to_decimal(self.limit_price)
        if limit_price <= 0:
            raise DomainError("지정가는 0보다 커야 합니다.")
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "confidence", require_confidence(self.confidence))


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
