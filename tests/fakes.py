"""여러 테스트가 공유하는 인메모리 어댑터."""

from decimal import Decimal

from trading_agent.application.errors import MarketDataUnavailable
from trading_agent.application.ports import (
    DailyBar,
    IndexPoint,
    IndicatorPoint,
    MarketDataGateway,
    MarketIndex,
    NewsItem,
    StockQuote,
    TreasuryMaturity,
)
from trading_agent.domain.models import AccountSnapshot, Position


class FakeGateway(MarketDataGateway):
    """사이클 테스트에서 사용하는 인메모리 시장 데이터."""

    def __init__(
        self, fail_account: bool = False, failing_quotes: frozenset[str] = frozenset()
    ) -> None:
        self.fail_account = fail_account
        self.failing_quotes = failing_quotes
        self.account_reads = 0

    def get_account(self) -> AccountSnapshot:
        self.account_reads += 1
        if self.fail_account:
            raise MarketDataUnavailable("account unavailable")
        return AccountSnapshot(Decimal("10000"), [Position("000660", "SK하이닉스", 1)])

    def get_stock_quote(self, code: str) -> StockQuote:
        if code in self.failing_quotes:
            raise MarketDataUnavailable(f"{code} quote unavailable")
        return StockQuote(code, "테스트 종목", Decimal("100"), Decimal("0"))

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        return []

    def get_moving_average(
        self, code: str, start_date: str, end_date: str, window: int
    ) -> list[IndicatorPoint]:
        return []

    def get_rsi(
        self, code: str, start_date: str, end_date: str, window: int = 14
    ) -> list[IndicatorPoint]:
        return []

    def get_rsi_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        rsi_window: int = 14,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        return []

    def get_macd(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
    ) -> list[IndicatorPoint]:
        return []

    def get_macd_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        return []

    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
        return []

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        return []

    def get_fear_and_greed_index(self) -> IndexPoint:
        return IndexPoint("20261029", Decimal("50"))

    def get_treasury_yield(
        self, maturity: TreasuryMaturity, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        return []
