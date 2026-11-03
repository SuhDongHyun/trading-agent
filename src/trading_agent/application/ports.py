"""외부 데이터 조회에 필요한 애플리케이션 포트."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from trading_agent.domain.models import AccountSnapshot


@dataclass(frozen=True, slots=True)
class StockQuote:
    """현재가, 선택적 시장 구분과 업종을 포함한 종목 가격 요약."""

    code: str
    name: str
    current_price: Decimal
    change_percent: Decimal
    market_name: str | None = None
    industry: str | None = None


@dataclass(frozen=True, slots=True)
class DailyBar:
    """종목의 일별 가격과 거래량, 선택적 OHLC 값."""

    date: str
    close_price: Decimal
    volume: int
    open_price: Decimal | None = None
    high_price: Decimal | None = None
    low_price: Decimal | None = None


@dataclass(frozen=True, slots=True)
class IndicatorPoint:
    """날짜별 지표 값과 선택적 매매 신호."""

    date: str
    value: Decimal
    signal: str | None = None


@dataclass(frozen=True, slots=True)
class NewsItem:
    """종목 뉴스의 제목과 게시 정보."""

    title: str
    source: str
    published_at: datetime
    key: str | None = None


@dataclass(frozen=True, slots=True)
class IndexPoint:
    """시장 지표 시계열의 단일 값."""

    date: str
    value: Decimal
    change_percent: Decimal | None = None


class MarketIndex(StrEnum):
    """trading-api가 제공하는 시장 시계열 이름."""

    KOSPI = "kospi-index"
    KOSDAQ = "kosdaq-index"
    SP500 = "sp500-index"
    VIX = "vix-index"
    VKOSPI = "vkospi-index"
    USD_KRW = "usd-krw-exchange-rate"


class TreasuryMaturity(StrEnum):
    """trading-api가 제공하는 국채 만기."""

    KOREA_3Y = "korea-3y"
    US_10Y = "us-10y"


class MarketDataGateway(ABC):
    """계좌, 종목 데이터, 시장 지표 조회 포트."""

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        """계좌 현금과 보유 종목을 조회한다."""

    @abstractmethod
    def get_stock_quote(self, code: str) -> StockQuote:
        """한 종목의 현재가를 조회한다."""

    @abstractmethod
    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        """종목 일봉을 조회한다."""

    @abstractmethod
    def get_moving_average(
        self, code: str, start_date: str, end_date: str, window: int
    ) -> list[IndicatorPoint]:
        """종목 이동평균 시계열을 조회한다."""

    @abstractmethod
    def get_rsi(
        self, code: str, start_date: str, end_date: str, window: int = 14
    ) -> list[IndicatorPoint]:
        """종목 RSI 시계열을 조회한다."""

    @abstractmethod
    def get_rsi_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        rsi_window: int = 14,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        """RSI signal EMA와 신호 시계열을 조회한다."""

    @abstractmethod
    def get_macd(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
    ) -> list[IndicatorPoint]:
        """MACD 시계열을 조회한다."""

    @abstractmethod
    def get_macd_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        """MACD signal EMA와 신호 시계열을 조회한다."""

    @abstractmethod
    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
        """한 종목의 날짜별 뉴스를 조회한다."""

    @abstractmethod
    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        """시장 지수나 환율 시계열을 조회한다."""

    @abstractmethod
    def get_fear_and_greed_index(self) -> IndexPoint:
        """최신 공포탐욕 지수를 조회한다."""

    @abstractmethod
    def get_treasury_yield(
        self, maturity: TreasuryMaturity, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        """국가와 만기별 국채 수익률을 조회한다."""


@dataclass(frozen=True, slots=True)
class ModelTask[T]:
    """모델에 맡길 작업의 역할 지시, 응답 스키마, 도메인 파서."""

    name: str
    system_prompt: str
    response_schema: dict[str, Any]
    parse: Callable[[str, str], T]


@dataclass(frozen=True, slots=True)
class ModelRun[T]:
    """모델 호출의 원문 출력과 검증 결과를 보존한다."""

    model_id: str
    stdout: str
    stderr: str
    exit_code: int | None
    duration_seconds: float
    response_text: str | None = None
    decision: T | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """정상 종료하고 도메인 판단까지 검증했는지 반환한다."""

        return self.exit_code == 0 and self.decision is not None and self.error is None


class DecisionModel[T](ABC):
    """프롬프트를 한 모델의 검증된 판단으로 변환하는 포트."""

    @abstractmethod
    def decide(self, prompt: str) -> ModelRun[T]:
        """판단 원문과 도메인 결과를 반환하고 실패 시 실행 정보를 예외에 담는다."""
