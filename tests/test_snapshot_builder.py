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
    ATR_WINDOW,
    DEFAULT_NEWS_LIMIT,
    DEFAULT_NEWS_LOOKBACK_TRADING_DAYS,
    MA_LONG_WINDOW,
    MA_SHORT_WINDOW,
    MACD_LONG_WINDOW,
    MACD_SHORT_WINDOW,
    MACD_SIGNAL_WINDOW,
    REALIZED_VOLATILITY_WINDOW,
    RSI_SIGNAL_WINDOW,
    RSI_WINDOW,
    SnapshotBuilder,
    calculate_20d_price_distances,
    calculate_52w_position,
    calculate_atr14_percent,
    calculate_deviation_percent,
    calculate_index_change_points,
    calculate_index_deviation_percent,
    calculate_index_return_percent,
    calculate_industry_relative_strength_20d_percent,
    calculate_realized_volatility_20d_percent,
    calculate_return_percent,
    calculate_universe_rank_percentile,
    calculate_volume_change_percent,
)
from trading_agent.domain.models import AccountSnapshot, Position, RuleType
from trading_agent.domain.risk import RiskPolicy
from trading_agent.domain.rules import generate_rule_proposals


class FakeMarketDataGateway(MarketDataGateway):
    """스냅샷 조립 테스트에 사용하는 메모리 데이터 포트."""

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.quote_calls: list[str] = []
        self.request_ranges: list[tuple[str, str]] = []
        self.market_requests: list[tuple[MarketIndex, str, str]] = []
        self.news_calls: list[tuple[str, str]] = []
        self.news_by_date: dict[str, list[NewsItem]] = {}

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
                Position("005930", "삼성전자", 2, profit_loss_rate=Decimal("10")),
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
        return [
            DailyBar(
                f"202610{i:02d}",
                Decimal(i),
                99 + i,
                high_price=Decimal(i + 2),
                low_price=Decimal(i),
            )
            for i in range(1, 26)
        ]

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
        self.news_calls.append((code, search_date))
        self._check("news")
        self._check(f"news:{search_date}")
        if search_date in self.news_by_date:
            return self.news_by_date[search_date]
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
        self.market_requests.append((index, start_date, end_date))
        if index is MarketIndex.SP500:
            return [
                IndexPoint("20261028", Decimal("100")),
                IndexPoint("20261029", Decimal("110")),
            ]
        if index in (MarketIndex.KOSPI, MarketIndex.KOSDAQ):
            weekdays = [
                datetime.strptime(end_date, "%Y%m%d") - timedelta(days=offset)
                for offset in range(100)
                if (
                    datetime.strptime(end_date, "%Y%m%d") - timedelta(days=offset)
                ).weekday()
                < 5
            ]
            dates = [day.strftime("%Y%m%d") for day in reversed(weekdays[:65])]
            base = Decimal(1000) if index is MarketIndex.KOSPI else Decimal(2000)
            points = [
                IndexPoint(
                    date,
                    base + Decimal(position),
                    Decimal("0.6") if position == len(dates) - 1 else None,
                )
                for position, date in enumerate(dates)
            ]
            return points
        if index is MarketIndex.VKOSPI:
            weekdays = [
                datetime.strptime(end_date, "%Y%m%d") - timedelta(days=offset)
                for offset in range(10)
                if (
                    datetime.strptime(end_date, "%Y%m%d") - timedelta(days=offset)
                ).weekday()
                < 5
            ]
            dates = [day.strftime("%Y%m%d") for day in reversed(weekdays[:6])]
            return [
                IndexPoint(date, Decimal("11.25") + Decimal(position))
                for position, date in enumerate(dates)
            ]
        value = Decimal("16.25") if index is MarketIndex.VIX else Decimal("2")
        return [IndexPoint(end_date, value)]

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
    assert snapshot.as_of_trading_date == "20261025"
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
    assert snapshot.stocks[0].atr14_percent == Decimal("3")
    assert snapshot.stocks[0].realized_volatility_20d_percent is not None
    assert snapshot.stocks[0].high_20d_distance_percent == (
        Decimal(100) - Decimal(27)
    ) / Decimal(27) * Decimal(100)
    assert snapshot.stocks[0].low_20d_distance_percent == (
        Decimal(100) - Decimal(6)
    ) / Decimal(6) * Decimal(100)
    assert snapshot.stocks[0].high_52w_position_percent == (
        Decimal(100) - Decimal(1)
    ) / Decimal(26) * Decimal(100)
    assert snapshot.stocks[0].high_52w_period_trading_days == 25
    assert [item.title for item in snapshot.stocks[0].news] == [
        "뉴스 0",
        "뉴스 1",
        "뉴스 2",
        "뉴스 3",
        "뉴스 4",
    ]
    assert snapshot.account.positions[0].current_price == Decimal("100")
    assert snapshot.account.positions[1].current_price == Decimal("200")
    assert snapshot.account.positions[0].average_price == Decimal("100") / Decimal(
        "1.1"
    )
    assert snapshot.account.total_asset_value == Decimal("1400")
    assert snapshot.market_context.kospi_change_percent == Decimal("0.6")
    assert snapshot.market_context.kospi_return_5d_percent is not None
    assert snapshot.market_context.kospi_return_20d_percent is not None
    assert snapshot.market_context.kospi_ma20_deviation_percent is not None
    assert snapshot.market_context.kospi_ma60_deviation_percent is not None
    assert snapshot.market_context.vkospi_change_5d_points == Decimal("5")
    assert snapshot.market_context.sp500_change_percent == Decimal("10")
    assert snapshot.market_context.volatility_index == Decimal("16.25")
    assert snapshot.market_context.fear_greed_index == Decimal("62.5")
    assert snapshot.market_context.korea_3y_yield == Decimal("3.25")
    assert snapshot.missing == []
    assert snapshot.market_session == "holiday"
    assert set(gateway.request_ranges) == {
        ("20251014", "20261029"),
        ("20260701", "20261029"),
        ("20260721", "20261029"),
        ("20261015", "20261029"),
    }
    assert (
        gateway.market_requests.count((MarketIndex.KOSPI, "20260721", "20261029")) == 1
    )
    assert (
        gateway.market_requests.count((MarketIndex.KOSDAQ, "20260721", "20261029")) == 1
    )
    assert all(
        start == "20261015"
        for index, start, _ in gateway.market_requests
        if index not in (MarketIndex.KOSPI, MarketIndex.KOSDAQ)
    )


def test_snapshot_builder_continues_after_optional_data_failures() -> None:
    gateway = FakeMarketDataGateway({"daily", "news", "market:vix"})
    snapshot = SnapshotBuilder(
        gateway,
        ["005930"],
        clock=lambda: datetime(2026, 10, 29, tzinfo=UTC),
    ).build()

    assert snapshot.stocks[0].current_price == Decimal("100")
    assert snapshot.stocks[0].return_5d_percent is None
    assert snapshot.stocks[0].news == []
    assert snapshot.market_context.volatility_index is None
    assert "005930:daily(unavailable)" in snapshot.missing
    assert "005930:news(20261029 unavailable)" in snapshot.missing
    assert "051910:daily(unavailable)" in snapshot.missing
    assert "051910:news(20261029 unavailable)" in snapshot.missing
    assert "market:vix(unavailable)" in snapshot.missing
    assert snapshot.account.total_asset_value == Decimal("1400")


def test_snapshot_builder_reports_empty_market_series_as_missing() -> None:
    class EmptyVkospiGateway(FakeMarketDataGateway):
        def get_market_index(
            self, index: MarketIndex, start_date: str, end_date: str
        ) -> list[IndexPoint]:
            if index is MarketIndex.VKOSPI:
                return []
            return super().get_market_index(index, start_date, end_date)

    snapshot = SnapshotBuilder(
        EmptyVkospiGateway(),
        ["005930"],
        clock=lambda: datetime(2026, 10, 29, tzinfo=UTC),
    ).build()

    assert snapshot.market_context.vkospi is None
    assert "market:vkospi(empty)" in snapshot.missing


def test_snapshot_builder_uses_kst_date_when_utc_clock_crosses_day_boundary() -> None:
    gateway = FakeMarketDataGateway()
    now = datetime(2026, 10, 29, 15, 30, tzinfo=UTC)

    snapshot = SnapshotBuilder(gateway, ["005930"], clock=lambda: now).build()

    assert snapshot.created_at == now
    assert set(gateway.request_ranges) == {
        ("20251015", "20261030"),
        ("20260702", "20261030"),
        ("20260722", "20261030"),
        ("20261016", "20261030"),
    }


def test_snapshot_builder_rejects_naive_clock_datetime() -> None:
    builder = SnapshotBuilder(
        FakeMarketDataGateway(),
        ["005930"],
        clock=lambda: datetime(2026, 10, 29, 15, 30),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        builder.build()


@pytest.mark.parametrize(
    ("price", "profit_loss_rate", "expected_rule"),
    [
        ("1000", "-10", RuleType.STOP_LOSS),
        ("1202", "20", RuleType.TAKE_PROFIT),
    ],
)
def test_snapshot_rate_triggers_rule_at_boundary_for_non_divisible_price(
    price: str,
    profit_loss_rate: str,
    expected_rule: RuleType,
) -> None:
    class AccountRateGateway(FakeMarketDataGateway):
        def get_account(self) -> AccountSnapshot:
            return AccountSnapshot(
                Decimal("10000"),
                [
                    Position(
                        "005930",
                        "삼성전자",
                        1,
                        profit_loss_rate=Decimal(profit_loss_rate),
                    )
                ],
            )

        def get_stock_quote(self, code: str) -> StockQuote:
            return StockQuote(code, "삼성전자", Decimal(price), Decimal("0"))

    snapshot = SnapshotBuilder(
        AccountRateGateway(),
        ["005930"],
        clock=lambda: datetime(2026, 10, 29, tzinfo=UTC),
    ).build()
    policy = RiskPolicy(
        max_position_weight=Decimal("1"),
        stop_loss_percent=Decimal("10"),
        take_profit_percent=Decimal("20"),
        allowed_codes={"005930"},
    )

    proposals = generate_rule_proposals(snapshot.account, policy)

    assert len(proposals) == 1
    assert proposals[0].rule_type is expected_rule


def test_research_fetches_today_and_recent_daily_dates_and_deduplicates() -> None:
    gateway = FakeMarketDataGateway()
    gateway.news_by_date = {
        "20261029": [
            NewsItem(
                "반복 기사",
                "source",
                datetime(2026, 10, 29, 12, tzinfo=UTC),
                "same-key",
            ),
            NewsItem("오늘 기사", "source", datetime(2026, 10, 29, 11, tzinfo=UTC)),
        ],
        "20261025": [
            NewsItem(
                "반복 기사",
                "source",
                datetime(2026, 10, 28, 12, tzinfo=UTC),
                "same-key",
            ),
            NewsItem("제목 중복", "source", datetime(2026, 10, 28, 11, tzinfo=UTC)),
            NewsItem(
                "제목 중복",
                "source",
                datetime(2026, 10, 28, 10, tzinfo=UTC),
                "another-key",
            ),
            NewsItem("이전 기사", "source", datetime(2026, 10, 28, 9, tzinfo=UTC)),
        ],
        "20261024": [
            NewsItem("24일 기사", "source", datetime(2026, 10, 24, 12, tzinfo=UTC))
        ],
        "20261023": [
            NewsItem("23일 기사", "source", datetime(2026, 10, 23, 12, tzinfo=UTC))
        ],
    }
    snapshot = SnapshotBuilder(
        gateway,
        [],
        clock=lambda: datetime(2026, 10, 29, 0, tzinfo=UTC),
    ).build_research(["005930"])

    assert gateway.news_calls == [
        ("005930", "20261029"),
        ("005930", "20261025"),
        ("005930", "20261024"),
        ("005930", "20261023"),
    ]
    assert [item.title for item in snapshot.stocks[0].news] == [
        "반복 기사",
        "오늘 기사",
        "제목 중복",
        "이전 기사",
        "24일 기사",
    ]
    assert [item.key for item in snapshot.stocks[0].news] == [
        "same-key",
        None,
        None,
        None,
        None,
    ]


def test_research_queries_news_once_when_latest_bar_is_today() -> None:
    class TodayBarGateway(FakeMarketDataGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            return [DailyBar(end_date, Decimal("100"), 10)]

    gateway = TodayBarGateway()
    SnapshotBuilder(
        gateway,
        [],
        clock=lambda: datetime(2026, 10, 29, 0, tzinfo=UTC),
    ).build_research(["005930"])

    assert gateway.news_calls == [("005930", "20261029")]


def test_ohlc_volatility_and_range_calculations() -> None:
    bars = [
        DailyBar(
            f"202601{day:02d}",
            Decimal("100"),
            1,
            high_price=Decimal("102"),
            low_price=Decimal("98"),
        )
        for day in range(1, 17)
    ]

    assert calculate_atr14_percent(bars, Decimal("200")) == Decimal("2")
    assert calculate_20d_price_distances(bars * 2, Decimal("100")) == (
        (Decimal("100") - Decimal("102")) / Decimal("102") * Decimal(100),
        (Decimal("100") - Decimal("98")) / Decimal("98") * Decimal(100),
    )
    position, days = calculate_52w_position(bars, Decimal("100"))
    assert position == Decimal("50")
    assert days == 16

    closes = [Decimal("100")]
    for _ in range(10):
        closes.extend((Decimal("101"), Decimal("100")))
    returns = [
        Decimal("1"),
        (Decimal("100") - Decimal("101")) / Decimal("101") * 100,
    ] * 10
    mean = sum(returns, Decimal(0)) / Decimal(20)
    expected = (
        sum(((item - mean) ** 2 for item in returns), Decimal(0)) / Decimal(20)
    ).sqrt()
    assert calculate_realized_volatility_20d_percent(closes) == expected


def test_wilder_atr_smooths_distinct_true_ranges_and_gap_candle() -> None:
    bars = [
        DailyBar(
            "20260101",
            Decimal("100"),
            1,
            high_price=Decimal("101"),
            low_price=Decimal("99"),
        )
    ]
    for day, true_range in enumerate(range(1, 15), start=2):
        half_range = Decimal(true_range) / Decimal(2)
        bars.append(
            DailyBar(
                f"202601{day:02d}",
                Decimal("100"),
                1,
                high_price=Decimal("100") + half_range,
                low_price=Decimal("100") - half_range,
            )
        )
    # 이 봉의 고저폭은 1이지만 전일 종가 대비 갭으로 TR은 20이다.
    bars.append(
        DailyBar(
            "20260116",
            Decimal("120"),
            1,
            high_price=Decimal("120"),
            low_price=Decimal("119"),
        )
    )
    bars.append(
        DailyBar(
            "20260117",
            Decimal("120"),
            1,
            high_price=Decimal("122"),
            low_price=Decimal("118"),
        )
    )

    # TR 1..14의 seed ATR은 7.5; 다음 TR 20과 4를 Wilder 식으로 갱신한다.
    seed_atr = sum((Decimal(value) for value in range(1, 15)), Decimal(0)) / 14
    after_gap = (seed_atr * 13 + Decimal("20")) / 14
    expected_atr = (after_gap * 13 + Decimal("4")) / 14
    expected_percent = expected_atr / Decimal("200") * Decimal(100)

    assert calculate_atr14_percent(bars, Decimal("200")) == expected_percent


def test_volatility_and_ohlc_calculations_report_missing_inputs() -> None:
    short_bars = [
        DailyBar("20260101", Decimal("100"), 1, high_price=None, low_price=None)
        for _ in range(14)
    ]
    closes = [Decimal(value) for value in range(1, 22)]

    assert calculate_atr14_percent(short_bars, Decimal("100")) is None
    assert calculate_realized_volatility_20d_percent(closes[:20]) is None
    assert calculate_20d_price_distances(short_bars, Decimal("100")) == (None, None)
    assert calculate_52w_position(short_bars, Decimal("100")) == (None, None)

    enough_bars = [
        DailyBar(
            f"202601{day:02d}",
            Decimal("100"),
            1,
            high_price=Decimal("101"),
            low_price=Decimal("99"),
        )
        for day in range(1, 22)
    ]
    missing_ohlc = enough_bars[:-1] + [
        DailyBar("20260121", Decimal("100"), 1, high_price=None, low_price=None)
    ]
    assert calculate_atr14_percent(missing_ohlc, Decimal("100")) is None
    assert calculate_20d_price_distances(missing_ohlc, Decimal("100")) == (None, None)
    assert calculate_52w_position(missing_ohlc, Decimal("100")) == (None, None)
    assert calculate_realized_volatility_20d_percent(closes) is not None


def test_zero_current_price_does_not_produce_normalized_metrics() -> None:
    bars = [
        DailyBar(
            f"202601{day:02d}",
            Decimal("100"),
            1,
            high_price=Decimal("101"),
            low_price=Decimal("99"),
        )
        for day in range(1, 22)
    ]

    assert calculate_atr14_percent(bars, Decimal("0")) is None
    assert calculate_20d_price_distances(bars, Decimal("0")) == (None, None)
    assert calculate_52w_position(bars, Decimal("0")) == (None, None)


def test_snapshot_builder_records_missing_reason_for_unavailable_ohlc() -> None:
    class MissingOhlcGateway(FakeMarketDataGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            return [
                DailyBar(f"202610{day:02d}", Decimal(day), 10) for day in range(1, 26)
            ]

    snapshot = SnapshotBuilder(
        MissingOhlcGateway(),
        ["005930"],
        clock=lambda: datetime(2026, 10, 29, tzinfo=UTC),
    ).build_research(["005930"])

    stock = snapshot.stocks[0]
    assert stock.atr14_percent is None
    assert stock.high_20d_distance_percent is None
    assert stock.low_20d_distance_percent is None
    assert stock.high_52w_position_percent is None
    assert stock.high_52w_period_trading_days is None
    assert "005930:atr14(OHLC unavailable)" in snapshot.missing
    assert "005930:high_20d_distance(OHLC unavailable)" in snapshot.missing
    assert "005930:low_20d_distance(OHLC unavailable)" in snapshot.missing
    assert "005930:high_52w_position(OHLC unavailable)" in snapshot.missing


def test_news_failure_on_one_date_does_not_discard_other_date_results() -> None:
    class PartialNewsGateway(FakeMarketDataGateway):
        def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
            self.news_calls.append((code, search_date))
            if search_date == "20261029":
                raise MarketDataUnavailable("news timeout", reason="timeout")
            return [
                NewsItem(
                    "이전 기사",
                    "source",
                    datetime(2026, 10, 28, tzinfo=UTC),
                )
            ]

    gateway = PartialNewsGateway()
    snapshot = SnapshotBuilder(
        gateway,
        [],
        clock=lambda: datetime(2026, 10, 29, 0, tzinfo=UTC),
    ).build_research(["005930"])

    assert [item.title for item in snapshot.stocks[0].news] == ["이전 기사"]
    assert snapshot.missing.count("005930:news(20261029 timeout)") == 1


def test_news_failures_keep_the_searched_date_and_each_failure_reason() -> None:
    class FailedNewsGateway(FakeMarketDataGateway):
        def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
            if search_date == "20261029":
                raise MarketDataUnavailable("timeout", reason="timeout")
            raise MarketDataUnavailable("server error", reason="http 503")

    snapshot = SnapshotBuilder(
        FailedNewsGateway(),
        [],
        clock=lambda: datetime(2026, 10, 29, 0, tzinfo=UTC),
    ).build_research(["005930"])

    assert "005930:news(20261029 timeout)" in snapshot.missing
    assert "005930:news(20261025 http 503)" in snapshot.missing


@pytest.mark.parametrize(
    ("created_at", "today"),
    [
        (datetime.fromisoformat("2026-10-29T11:00:00+09:00"), "20261029"),
        (datetime.fromisoformat("2026-10-29T16:00:00+09:00"), "20261029"),
        (datetime.fromisoformat("2026-10-31T10:00:00+09:00"), "20261031"),
    ],
)
def test_news_dates_include_today_and_recent_available_bars(
    created_at: datetime, today: str
) -> None:
    gateway = FakeMarketDataGateway()

    SnapshotBuilder(gateway, [], clock=lambda: created_at).build_research(["005930"])

    assert [date for _, date in gateway.news_calls] == [
        today,
        "20261025",
        "20261024",
        "20261023",
    ]


@pytest.mark.parametrize(
    "created_at",
    [
        datetime.fromisoformat("2026-10-29T11:00:00+09:00"),
        datetime.fromisoformat("2026-10-29T16:00:00+09:00"),
    ],
)
def test_live_news_dates_include_today_and_n_previous_days_when_today_bar_exists(
    created_at: datetime,
) -> None:
    class TodayAndPreviousBars(FakeMarketDataGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            return [
                DailyBar(f"202610{day}", Decimal(day), day) for day in range(26, 30)
            ]

    gateway = TodayAndPreviousBars()

    SnapshotBuilder(gateway, [], clock=lambda: created_at).build_research(["005930"])

    assert [date for _, date in gateway.news_calls] == [
        "20261029",
        "20261028",
        "20261027",
        "20261026",
    ]


def test_news_dates_deduplicate_daily_rows_and_work_without_daily_bars() -> None:
    class DuplicateDailyGateway(FakeMarketDataGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            return [
                DailyBar("20261028", Decimal("100"), 1),
                DailyBar("20261028", Decimal("101"), 2),
                DailyBar("20261027", Decimal("99"), 3),
            ]

    now = datetime.fromisoformat("2026-10-29T10:00:00+09:00")
    duplicate_gateway = DuplicateDailyGateway()
    SnapshotBuilder(duplicate_gateway, [], clock=lambda: now).build_research(["005930"])
    assert [date for _, date in duplicate_gateway.news_calls] == [
        "20261029",
        "20261028",
        "20261027",
    ]

    no_bars = FakeMarketDataGateway({"daily"})
    SnapshotBuilder(no_bars, [], clock=lambda: now).build_research(["005930"])
    assert [date for _, date in no_bars.news_calls] == ["20261029"]


def test_news_items_normalize_times_and_include_age_and_after_close() -> None:
    gateway = FakeMarketDataGateway()
    gateway.news_by_date["20261029"] = [
        NewsItem(
            "장 마감 시각",
            "source",
            datetime.fromisoformat("2026-10-29T06:30:00+00:00"),
        ),
        NewsItem("장 마감 뒤", "source", datetime(2026, 10, 29, 15, 31)),
    ]
    now = datetime.fromisoformat("2026-10-29T15:40:00+09:00")

    snapshot = SnapshotBuilder(
        gateway,
        [],
        clock=lambda: now,
        news_lookback_trading_days=1,
        news_limit=2,
    ).build_research(["005930"])

    assert [item.title for item in snapshot.stocks[0].news] == [
        "장 마감 뒤",
        "장 마감 시각",
    ]
    assert (
        snapshot.stocks[0].news[0].published_at.isoformat()
        == "2026-10-29T15:31:00+09:00"
    )
    assert snapshot.stocks[0].news[0].age_hours == 0.2
    assert snapshot.stocks[0].news[0].after_close is True
    assert (
        snapshot.stocks[0].news[1].published_at.isoformat()
        == "2026-10-29T15:30:00+09:00"
    )
    assert snapshot.stocks[0].news[1].age_hours == 0.2
    assert snapshot.stocks[0].news[1].after_close is False


def test_data_basis_matches_indicator_constants_and_news_settings() -> None:
    builder = SnapshotBuilder(
        FakeMarketDataGateway(),
        [],
        clock=lambda: datetime.fromisoformat("2026-10-29T10:00:00+09:00"),
        news_lookback_trading_days=4,
    )
    snapshot = builder.build_research(["005930"])
    market_snapshot = SnapshotBuilder(
        FakeMarketDataGateway(),
        ["005930"],
        clock=lambda: datetime.fromisoformat("2026-10-29T10:00:00+09:00"),
        news_lookback_trading_days=4,
    ).build()

    assert snapshot.data_basis is not None
    assert snapshot.data_basis.price_basis == "adjusted"
    assert snapshot.data_basis.rsi == RSI_WINDOW
    assert snapshot.data_basis.rsi_signal_ema == RSI_SIGNAL_WINDOW
    assert snapshot.data_basis.macd == (MACD_SHORT_WINDOW, MACD_LONG_WINDOW)
    assert snapshot.data_basis.macd_signal_ema == MACD_SIGNAL_WINDOW
    assert snapshot.data_basis.ma == (MA_SHORT_WINDOW, MA_LONG_WINDOW)
    assert snapshot.data_basis.atr == ATR_WINDOW
    assert snapshot.data_basis.realized_volatility == REALIZED_VOLATILITY_WINDOW
    assert snapshot.data_basis.news_lookback_trading_days == 4
    assert market_snapshot.data_basis == snapshot.data_basis
    assert DEFAULT_NEWS_LOOKBACK_TRADING_DAYS == 3
    assert DEFAULT_NEWS_LIMIT == 5


@pytest.mark.parametrize(
    ("news_limit", "news_lookback_trading_days"),
    [(0, 3), (21, 3), (5, 0), (5, 11)],
)
def test_snapshot_builder_rejects_out_of_range_news_settings(
    news_limit: int, news_lookback_trading_days: int
) -> None:
    with pytest.raises(ValueError):
        SnapshotBuilder(
            FakeMarketDataGateway(),
            [],
            news_limit=news_limit,
            news_lookback_trading_days=news_lookback_trading_days,
        )


@pytest.mark.parametrize(
    ("now", "as_of", "expected"),
    [
        (datetime(2026, 10, 31, 10, tzinfo=UTC), "20261030", "holiday"),
        (datetime(2026, 10, 29, 23, 30, tzinfo=UTC), None, "pre_open"),
        (datetime(2026, 10, 29, 1, tzinfo=UTC), None, "open"),
        (datetime(2026, 10, 29, 1, tzinfo=UTC), "20261029", "open"),
        (datetime(2026, 10, 29, 7, tzinfo=UTC), None, "closed"),
        (datetime(2026, 10, 29, 7, tzinfo=UTC), "20261028", "holiday"),
    ],
)
def test_snapshot_exposes_kst_market_session(
    now: datetime, as_of: str | None, expected: str
) -> None:
    class BarDateGateway(FakeMarketDataGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            return [] if as_of is None else [DailyBar(as_of, Decimal("100"), 10)]

    snapshot = SnapshotBuilder(BarDateGateway(), ["005930"], clock=lambda: now).build()

    assert snapshot.as_of_trading_date == as_of
    assert snapshot.market_session == expected


@pytest.mark.parametrize("failure", ["account", "quote"])
def test_snapshot_builder_fails_closed_for_required_data(failure: str) -> None:
    gateway = FakeMarketDataGateway({failure})
    builder = SnapshotBuilder(gateway, ["005930"])

    expected = "005930:quote(unavailable)" if failure == "quote" else None
    with pytest.raises(MarketDataUnavailable) as error:
        builder.build()
    if expected is not None:
        assert str(error.value) == expected


def test_research_snapshot_records_failed_quote_reason_and_continues() -> None:
    class PartialQuoteGateway(FakeMarketDataGateway):
        def get_stock_quote(self, code: str) -> StockQuote:
            if code == "005930":
                raise MarketDataUnavailable("unavailable", reason="http 503")
            return super().get_stock_quote(code)

    snapshot = SnapshotBuilder(
        PartialQuoteGateway(),
        [],
        clock=lambda: datetime(2026, 10, 29, tzinfo=UTC),
    ).build_research(["005930", "000660"])

    assert [stock.code for stock in snapshot.stocks] == ["000660"]
    assert "005930:quote(http 503)" in snapshot.missing


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


def test_market_index_calculations_cover_returns_deviations_and_history_limits() -> (
    None
):
    points = [IndexPoint(f"202601{day:02}", Decimal(100 + day)) for day in range(1, 62)]

    assert calculate_index_return_percent(points, 5) == (
        Decimal(5) / Decimal(156) * Decimal(100)
    )
    assert calculate_index_return_percent(points, 20) == (
        Decimal(20) / Decimal(141) * Decimal(100)
    )
    assert calculate_index_deviation_percent(points, 20) == (
        (Decimal(161) - Decimal("151.5")) / Decimal("151.5") * Decimal(100)
    )
    assert calculate_index_deviation_percent(points, 60) is not None
    assert calculate_index_return_percent(points[-5:], 5) is None
    assert calculate_index_return_percent(points[-20:], 20) is None
    assert calculate_index_deviation_percent(points[-19:], 20) is None
    assert calculate_index_deviation_percent(points[-59:], 60) is None
    assert (
        calculate_index_return_percent(
            [IndexPoint("20260101", Decimal(0)), IndexPoint("20260102", Decimal(1))], 1
        )
        is None
    )


def test_vkospi_five_trading_day_change_uses_points_and_requires_six_values() -> None:
    points = [IndexPoint(f"202601{day:02}", Decimal(day)) for day in range(1, 7)]

    assert calculate_index_change_points(points, 5) == Decimal(5)
    assert calculate_index_change_points(points[:-1], 5) is None


def test_relative_strength_rank_and_industry_helpers_define_ties_and_no_peers() -> None:
    assert calculate_universe_rank_percentile(Decimal(10), [Decimal(10)]) is None
    assert calculate_universe_rank_percentile(
        Decimal(10), [Decimal(10), Decimal(10)]
    ) == Decimal(50)
    assert calculate_universe_rank_percentile(
        Decimal(10), [Decimal(5), Decimal(10), Decimal(10), Decimal(20), None]
    ) == Decimal(50)
    assert calculate_universe_rank_percentile(
        Decimal(5), [None, Decimal(5), Decimal(10)]
    ) == Decimal(0)
    assert calculate_universe_rank_percentile(None, [Decimal(5), Decimal(10)]) is None
    assert calculate_industry_relative_strength_20d_percent(
        Decimal(10), "전자", [("전자", Decimal(4)), ("전자", Decimal(6))]
    ) == Decimal(5)
    assert (
        calculate_industry_relative_strength_20d_percent(Decimal(10), "전자", [])
        is None
    )
    assert (
        calculate_industry_relative_strength_20d_percent(
            Decimal(10), None, [("전자", Decimal(4))]
        )
        is None
    )


def test_snapshot_builder_enriches_run_and_recommend_snapshots_after_collection() -> (
    None
):
    class RelativeStrengthGateway(FakeMarketDataGateway):
        quote_data = {
            "005930": ("KOSPI200", "전자", 2),
            "000660": ("KSQ150", "전자", 1),
            "051910": ("KOSPI", "화학", 0.5),
            "086520": (None, None, 3),
        }

        def get_stock_quote(self, code: str) -> StockQuote:
            market_name, industry, _ = self.quote_data[code]
            return StockQuote(
                code,
                f"종목-{code}",
                Decimal(100),
                Decimal(0),
                market_name=market_name,
                industry=industry,
            )

        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            _, _, increment = self.quote_data[code]
            return [
                DailyBar(
                    f"202610{day:02}",
                    Decimal(100) + Decimal(str(increment)) * Decimal(day),
                    100,
                    high_price=Decimal(102) + Decimal(str(increment)) * Decimal(day),
                    low_price=Decimal(98) + Decimal(str(increment)) * Decimal(day),
                )
                for day in range(1, 26)
            ]

    gateway = RelativeStrengthGateway()
    now = datetime(2026, 10, 29, tzinfo=UTC)
    builder = SnapshotBuilder(gateway, ["005930", "000660"], clock=lambda: now)

    run_snapshot = builder.build()
    recommend_snapshot = builder.build_research(
        ["005930", "000660", "051910", "086520"]
    )

    for snapshot in (run_snapshot, recommend_snapshot):
        stocks = {stock.code: stock for stock in snapshot.stocks}
        assert stocks["005930"].industry == "전자"
        assert stocks["005930"].relative_strength_5d_percent is not None
        assert stocks["005930"].relative_strength_20d_percent is not None
        assert stocks["005930"].universe_rank_percentile_20d is not None
        assert stocks["005930"].industry_relative_strength_20d_percent is not None
        assert stocks["000660"].relative_strength_20d_percent is not None
        assert stocks["000660"].industry_relative_strength_20d_percent is not None
        assert stocks["051910"].relative_strength_20d_percent is not None
    recommend_stocks = {stock.code: stock for stock in recommend_snapshot.stocks}
    assert recommend_stocks["086520"].relative_strength_5d_percent is None
    assert recommend_stocks["086520"].relative_strength_20d_percent is None
    assert recommend_stocks["086520"].industry_relative_strength_20d_percent is None


def test_insufficient_market_history_is_reported_for_new_fields() -> None:
    class ShortMarketGateway(FakeMarketDataGateway):
        def get_market_index(
            self, index: MarketIndex, start_date: str, end_date: str
        ) -> list[IndexPoint]:
            if index in (MarketIndex.KOSPI, MarketIndex.KOSDAQ, MarketIndex.VKOSPI):
                return [
                    IndexPoint(f"202610{day:02}", Decimal(day)) for day in range(1, 6)
                ]
            return super().get_market_index(index, start_date, end_date)

    snapshot = SnapshotBuilder(
        ShortMarketGateway(),
        [],
        clock=lambda: datetime(2026, 10, 29, tzinfo=UTC),
    ).build_research(["005930"])

    assert snapshot.market_context.kospi_return_5d_percent is None
    assert snapshot.market_context.kospi_return_20d_percent is None
    assert snapshot.market_context.kospi_ma20_deviation_percent is None
    assert snapshot.market_context.kospi_ma60_deviation_percent is None
    assert snapshot.market_context.vkospi_change_5d_points is None
    assert "market:kospi_return_5d(insufficient_history)" in snapshot.missing
    assert "market:kospi_return_20d(insufficient_history)" in snapshot.missing
    assert "market:kospi_ma60_deviation(insufficient_history)" in snapshot.missing
    assert "market:vkospi_change_5d_points(insufficient_history)" in snapshot.missing
