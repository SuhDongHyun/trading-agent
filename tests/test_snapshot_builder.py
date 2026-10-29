from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

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
from trading_agent.application.snapshot_builder import (
    SnapshotBuilder,
    calculate_deviation_percent,
    calculate_return_percent,
    calculate_volume_change_percent,
)
from trading_agent.domain.models import AccountSnapshot, Position


class FakeMarketDataGateway(MarketDataGateway):
    """스냅샷 조립 테스트에 사용하는 메모리 데이터 포트."""

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.quote_calls: list[str] = []
        self.request_ranges: list[tuple[str, str]] = []

    def _check(self, key: str) -> None:
        if key in self.failures:
            raise MarketDataUnavailable(key)

    def _range(self, start_date: str, end_date: str) -> None:
        self.request_ranges.append((start_date, end_date))

    def get_account(self) -> AccountSnapshot:
        self._check("account")
        return AccountSnapshot(
            cash=Decimal("1000"),
            positions=[
                Position("005930", "삼성전자", 2),
                Position("051910", "LG화학", 1),
            ],
        )

    def get_stock_quote(self, code: str) -> StockQuote:
        self._check("quote")
        self.quote_calls.append(code)
        price = Decimal("100") if code == "005930" else Decimal("200")
        return StockQuote(code, f"종목-{code}", price, Decimal("1.25"))

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        self._check("daily")
        self._range(start_date, end_date)
        return [DailyBar(f"202610{i:02d}", Decimal(i), 99 + i) for i in range(1, 26)]

    def get_moving_average(
        self, code: str, start_date: str, end_date: str, window: int
    ) -> list[IndicatorPoint]:
        self._check(f"ma{window}")
        self._range(start_date, end_date)
        return [IndicatorPoint("20261029", Decimal(window))]

    def get_rsi(
        self, code: str, start_date: str, end_date: str, window: int = 14
    ) -> list[IndicatorPoint]:
        self._check("rsi")
        self._range(start_date, end_date)
        return [IndicatorPoint("20261029", Decimal("55.5"))]

    def get_rsi_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        rsi_window: int = 14,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        self._check("rsi_signal")
        self._range(start_date, end_date)
        return [IndicatorPoint("20261029", Decimal("54.5"), "buy")]

    def get_macd(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
    ) -> list[IndicatorPoint]:
        self._check("macd")
        self._range(start_date, end_date)
        return [IndicatorPoint("20261029", Decimal("1.25"))]

    def get_macd_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        self._check("macd_signal")
        self._range(start_date, end_date)
        return [IndicatorPoint("20261029", Decimal("1"), "neutral")]

    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
        self._check("news")
        return [
            NewsItem(
                title=f"뉴스 {number}",
                source="연합뉴스",
                published_at=datetime(2026, 10, 29, tzinfo=UTC)
                - timedelta(minutes=number),
            )
            for number in range(7)
        ]

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        self._check(f"market:{index.name.lower()}")
        self._range(start_date, end_date)
        if index is MarketIndex.SP500:
            return [
                IndexPoint("20261028", Decimal("100")),
                IndexPoint("20261029", Decimal("110")),
            ]
        value = Decimal("16.25") if index is MarketIndex.VIX else Decimal("2")
        change = (
            Decimal("0.6") if index in (MarketIndex.KOSPI, MarketIndex.KOSDAQ) else None
        )
        return [IndexPoint("20261029", value, change)]

    def get_fear_and_greed_index(self) -> IndexPoint:
        self._check("market:fear_and_greed")
        return IndexPoint("20261029", Decimal("62.5"))

    def get_treasury_yield(
        self, maturity: TreasuryMaturity, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        self._check(f"market:{maturity.name.lower()}")
        self._range(start_date, end_date)
        return [IndexPoint("20261029", Decimal("3.25"))]


def test_snapshot_builder_merges_watchlist_and_holdings_and_summarizes_data() -> None:
    gateway = FakeMarketDataGateway()
    now = datetime(2026, 10, 29, 11, 30, tzinfo=UTC)

    snapshot = SnapshotBuilder(
        gateway,
        ["005930", "000660", "005930"],
        clock=lambda: now,
    ).build()

    assert gateway.quote_calls == ["005930", "000660", "051910"]
    assert [stock.code for stock in snapshot.stocks] == [
        "005930",
        "000660",
        "051910",
    ]
    assert snapshot.created_at == now
    assert snapshot.stocks[0].return_5d_percent == Decimal("25")
    assert snapshot.stocks[0].return_20d_percent == Decimal("400")
    assert snapshot.stocks[0].ma20_deviation_percent == Decimal("400")
    assert snapshot.stocks[0].ma60_deviation_percent == Decimal(
        "66.66666666666666666666666667"
    )
    assert snapshot.stocks[0].rsi == Decimal("55.5")
    assert snapshot.stocks[0].rsi_signal == "buy"
    assert snapshot.stocks[0].macd == Decimal("1.25")
    assert snapshot.stocks[0].macd_signal_value == Decimal("1")
    assert snapshot.stocks[0].macd_signal == "neutral"
    assert snapshot.stocks[0].news_headlines == [
        "뉴스 0",
        "뉴스 1",
        "뉴스 2",
        "뉴스 3",
        "뉴스 4",
    ]
    assert snapshot.account.positions[0].current_price == Decimal("100")
    assert snapshot.account.positions[1].current_price == Decimal("200")
    assert snapshot.account.total_asset_value == Decimal("1400")
    assert snapshot.market_context.kospi_change_percent == Decimal("0.6")
    assert snapshot.market_context.sp500_change_percent == Decimal("10")
    assert snapshot.market_context.volatility_index == Decimal("16.25")
    assert snapshot.market_context.fear_greed_index == Decimal("62.5")
    assert snapshot.market_context.korea_3y_yield == Decimal("3.25")
    assert snapshot.missing == []
    assert set(gateway.request_ranges) == {("20260701", "20261029")}


def test_snapshot_builder_continues_after_optional_data_failures() -> None:
    gateway = FakeMarketDataGateway({"daily", "news", "market:vix"})
    snapshot = SnapshotBuilder(
        gateway,
        ["005930"],
        clock=lambda: datetime(2026, 10, 29, tzinfo=UTC),
    ).build()

    assert snapshot.stocks[0].current_price == Decimal("100")
    assert snapshot.stocks[0].return_5d_percent is None
    assert snapshot.stocks[0].news_headlines is None
    assert snapshot.market_context.volatility_index is None
    assert "005930:daily" in snapshot.missing
    assert "005930:news" in snapshot.missing
    assert "051910:daily" in snapshot.missing
    assert "051910:news" in snapshot.missing
    assert "market:vix" in snapshot.missing
    assert snapshot.account.total_asset_value == Decimal("1400")


@pytest.mark.parametrize("failure", ["account", "quote"])
def test_snapshot_builder_fails_closed_for_required_data(failure: str) -> None:
    gateway = FakeMarketDataGateway({failure})
    builder = SnapshotBuilder(gateway, ["005930"])

    with pytest.raises(MarketDataUnavailable):
        builder.build()


def test_calculation_helpers_cover_boundaries() -> None:
    assert calculate_return_percent(
        [Decimal("100"), Decimal("110")], periods=1
    ) == Decimal("10")
    assert calculate_return_percent([Decimal("100")], periods=1) is None
    assert calculate_return_percent([Decimal("0"), Decimal("1")], periods=1) is None
    assert calculate_deviation_percent(Decimal("110"), Decimal("100")) == Decimal("10")
    assert calculate_deviation_percent(Decimal("100"), Decimal("0")) is None
    assert calculate_deviation_percent(Decimal("100"), None) is None
    assert calculate_volume_change_percent([10, 10, 15], window=2) == Decimal("50")
    assert calculate_volume_change_percent([10, 10], window=2) is None
    assert calculate_volume_change_percent([0, 0, 10], window=2) is None

    with pytest.raises(ValueError):
        calculate_return_percent([Decimal("1")], periods=0)
    with pytest.raises(ValueError):
        calculate_volume_change_percent([1], window=0)
