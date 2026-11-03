from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeGateway

from trading_agent.application.backtest import (
    BacktestOptions,
    MaskedSnapshotProvider,
    MaskingMap,
    RunBacktest,
    choose_dates,
    last_safe_date,
    mask_snapshot,
    restore_recommendation,
)
from trading_agent.application.consensus_service import RecommendationConsensusService
from trading_agent.application.errors import MarketDataUnavailable
from trading_agent.application.ports import (
    DailyBar,
    DecisionModel,
    IndexPoint,
    IndicatorPoint,
    MarketIndex,
    ModelRun,
    NewsItem,
    StockQuote,
    TreasuryMaturity,
)
from trading_agent.application.recommend_stocks import RecommendStocks
from trading_agent.application.snapshot_builder import (
    LOOKBACK_CALENDAR_DAYS,
    MARKET_INDEX_LOOKBACK_CALENDAR_DAYS,
    SnapshotBuilder,
)
from trading_agent.domain.models import (
    AccountSnapshot,
    Action,
    MarketContext,
    MarketSnapshot,
    ResearchSnapshot,
    StockSnapshot,
)
from trading_agent.domain.recommendation import ModelRecommendation, StockRecommendation
from trading_agent.domain.risk import RiskPolicy
from trading_agent.infrastructure.backtest_result_store import FileBacktestResultStore
from trading_agent.infrastructure.config import load_settings
from trading_agent.infrastructure.point_in_time_gateway import (
    PointInTimeGateway,
    SharedDailyCache,
)
from trading_agent.infrastructure.response_cache import (
    ResponseCache,
    atomic_write_text,
)
from trading_agent.interface.cli import build_parser, main


class HistoricalGateway(FakeGateway):
    def __init__(self) -> None:
        super().__init__()
        self.daily_reads = 0
        self.requested_ends: list[tuple[str, str]] = []
        self.requested_market_ranges: list[tuple[str, str, str]] = []
        self.news_calls: list[str] = []

    def get_stock_quote(self, code: str) -> StockQuote:
        return StockQuote(
            code,
            "현재 이름",
            Decimal("999"),
            Decimal("99"),
            market_name="KOSDAQ",
            industry="전자",
        )

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        self.daily_reads += 1
        return [
            DailyBar(
                "20260914",
                Decimal("100"),
                1,
                high_price=Decimal("101"),
                low_price=Decimal("99"),
            ),
            DailyBar(
                "20260915",
                Decimal("110"),
                2,
                high_price=Decimal("111"),
                low_price=Decimal("109"),
            ),
            DailyBar(
                "20260916",
                Decimal("120"),
                3,
                high_price=Decimal("121"),
                low_price=Decimal("119"),
            ),
        ]

    def get_rsi(
        self, code: str, start_date: str, end_date: str, window: int = 14
    ) -> list[IndicatorPoint]:
        self.requested_ends.append(("rsi", end_date))
        return [
            IndicatorPoint("20260915", Decimal("50")),
            IndicatorPoint("20260916", Decimal("60")),
        ]

    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
        self.news_calls.append(search_date)
        return [
            NewsItem(
                "이전", "source", datetime.fromisoformat("2026-09-15T15:39:00+09:00")
            ),
            NewsItem(
                "이후", "source", datetime.fromisoformat("2026-09-15T15:41:00+09:00")
            ),
            NewsItem("naive 이후", "source", datetime(2026, 9, 15, 15, 41)),
        ]

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        self.requested_ends.append((index.value, end_date))
        self.requested_market_ranges.append((index.value, start_date, end_date))
        return [
            IndexPoint("20260914", Decimal("10")),
            IndexPoint("20260915", Decimal("11")),
        ]

    def get_treasury_yield(
        self, maturity: TreasuryMaturity, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        self.requested_ends.append((maturity.value, end_date))
        return [
            IndexPoint("20260914", Decimal("3")),
            IndexPoint("20260915", Decimal("4")),
        ]


def test_point_in_time_gateway_cuts_daily_rows_and_shares_ticker_cache() -> None:
    inner = HistoricalGateway()
    shared = SharedDailyCache()
    first = PointInTimeGateway(
        inner, "20260915", "20260901", "20260916", daily_cache=shared
    )
    second = PointInTimeGateway(
        inner, "20260916", "20260901", "20260916", daily_cache=shared
    )

    assert [
        bar.date for bar in first.get_daily_bars("005930", "20260901", "20260916")
    ] == [
        "20260914",
        "20260915",
    ]
    assert [
        bar.date for bar in second.get_daily_bars("005930", "20260901", "20260916")
    ] == [
        "20260914",
        "20260915",
        "20260916",
    ]
    assert first.leakage["daily_rows"] == 1
    assert second.leakage["daily_rows"] == 0
    assert first.get_daily_bars("005930", "20260901", "20260916")[
        -1
    ].high_price == Decimal("111")
    assert first.get_daily_bars("005930", "20260901", "20260916")[
        -1
    ].low_price == Decimal("109")
    assert inner.daily_reads == 1


def test_point_in_time_quote_uses_decision_close_and_previous_close() -> None:
    gateway = PointInTimeGateway(
        HistoricalGateway(), "20260915", "20260901", "20260916"
    )

    quote = gateway.get_stock_quote("005930")

    assert quote.name == "현재 이름"
    assert quote.current_price == Decimal("110")
    assert quote.change_percent == Decimal("10")
    assert quote.market_name == "KOSDAQ"
    assert quote.industry == "전자"


def test_point_in_time_quote_requires_decision_day_bar() -> None:
    class NoDecisionBar(HistoricalGateway):
        def get_daily_bars(self, code: str, start_date: str, end_date: str):
            return [DailyBar("20260914", Decimal("100"), 1)]

    gateway = PointInTimeGateway(NoDecisionBar(), "20260915", "20260901", "20260916")
    with pytest.raises(MarketDataUnavailable, match="기준일 일봉"):
        gateway.get_stock_quote("005930")


def test_point_in_time_indicators_clamp_request_and_count_future_rows() -> None:
    inner = HistoricalGateway()
    gateway = PointInTimeGateway(inner, "20260915", "20260901", "20260916")

    points = gateway.get_rsi("005930", "20260901", "20260916")

    assert [point.date for point in points] == ["20260915"]
    assert inner.requested_ends == [("rsi", "20260915")]
    assert gateway.leakage["rsi_rows"] == 1


def test_point_in_time_news_hides_later_search_dates_and_after_clock_articles() -> None:
    inner = HistoricalGateway()
    gateway = PointInTimeGateway(
        inner, "20260915", "20260901", "20260916", decision_time=time(15, 40)
    )

    articles = gateway.get_stock_news("005930", "20260915")
    assert [item.title for item in articles] == ["이전"]
    assert gateway.get_stock_news("005930", "20260916") == []
    assert inner.news_calls == ["20260915"]
    assert gateway.leakage["news_articles"] == 2


def test_point_in_time_market_cutoffs_include_korea_and_exclude_us_decision_day() -> (
    None
):
    inner = HistoricalGateway()
    gateway = PointInTimeGateway(inner, "20260915", "20260901", "20260916")

    korea = gateway.get_market_index(MarketIndex.KOSPI, "20260901", "20260916")
    us = gateway.get_market_index(MarketIndex.SP500, "20260901", "20260916")
    korea_yield = gateway.get_treasury_yield(
        TreasuryMaturity.KOREA_3Y, "20260901", "20260916"
    )
    us_yield = gateway.get_treasury_yield(
        TreasuryMaturity.US_10Y, "20260901", "20260916"
    )

    assert [item.date for item in korea] == ["20260914", "20260915"]
    assert [item.date for item in us] == ["20260914"]
    assert [item.date for item in korea_yield] == ["20260914", "20260915"]
    assert [item.date for item in us_yield] == ["20260914"]
    assert ("sp500-index", "20260914") in inner.requested_ends
    assert ("us-10y", "20260914") in inner.requested_ends


def test_point_in_time_fear_greed_is_missing() -> None:
    gateway = PointInTimeGateway(
        HistoricalGateway(), "20260915", "20260901", "20260916"
    )
    with pytest.raises(MarketDataUnavailable, match="과거 시점") as error:
        gateway.get_fear_and_greed_index()
    assert error.value.reason == "not point-in-time"


def test_date_selection_is_seeded_and_safe_to_date_uses_horizon() -> None:
    dates = [f"202609{day:02}" for day in range(1, 31)]
    assert choose_dates(dates, dates[0], dates[-1], 5, 42) == choose_dates(
        dates, dates[0], dates[-1], 5, 42
    )
    assert last_safe_date(dates, 5) == "20260925"
    assert choose_dates(dates, dates[0], dates[-1], "all", 42) == dates


def test_backtest_run_defaults_to_ten_dates_and_all_is_explicit() -> None:
    from trading_agent.interface.bootstrap import _parse_dates_count

    args = build_parser().parse_args(["backtest", "run"])

    assert args.dates is None
    assert _parse_dates_count(args.dates) == 10
    assert _parse_dates_count("all") == "all"
    explicit_args = build_parser().parse_args(["backtest", "run", "--dates", "10"])
    assert explicit_args.dates == "10"


def test_backtest_pit_gateway_range_covers_extended_daily_lookback(
    tmp_path: Path,
) -> None:
    options = make_options(tmp_path / "extended-lookback")
    ranges: list[tuple[str, str]] = []

    assert make_runner(options, gateway_ranges=ranges).run() == 0

    expected_start = (
        datetime.strptime("20260915", "%Y%m%d")
        - timedelta(days=LOOKBACK_CALENDAR_DAYS + 14)
    ).strftime("%Y%m%d")
    assert ranges == [(expected_start, "20260915")]


def test_snapshot_builder_passes_extended_korean_index_lookback_through_pit() -> None:
    inner = HistoricalGateway()
    decision_date = "20260915"
    gateway = PointInTimeGateway(
        inner,
        decision_date,
        "20250826",
        decision_date,
    )
    builder = SnapshotBuilder(
        gateway,
        [],
        clock=lambda: datetime.fromisoformat("2026-09-15T15:40:00+09:00"),
    )

    builder.build_research(["005930"])

    index_start = (
        datetime.strptime(decision_date, "%Y%m%d")
        - timedelta(days=MARKET_INDEX_LOOKBACK_CALENDAR_DAYS)
    ).strftime("%Y%m%d")
    market_start = (
        datetime.strptime(decision_date, "%Y%m%d") - timedelta(days=14)
    ).strftime("%Y%m%d")
    assert (MarketIndex.KOSPI.value, index_start, decision_date) in (
        inner.requested_market_ranges
    )
    assert (MarketIndex.KOSDAQ.value, index_start, decision_date) in (
        inner.requested_market_ranges
    )
    assert (MarketIndex.VKOSPI.value, market_start, decision_date) in (
        inner.requested_market_ranges
    )


def test_implicit_dates_count_is_capped_at_available_dates(tmp_path: Path) -> None:
    options = make_options(
        tmp_path / "implicit-short-range",
        dates=10,
        dates_explicit=False,
        explicit_dates=None,
        snapshot_only=True,
        use_default_explicit_date=False,
    )
    runner = make_runner(
        options,
        calendar_dates=["20260914", "20260915"],
    )

    assert runner.run() == 0
    manifest = FileBacktestResultStore(options.output_dir).read_manifest()
    assert manifest is not None
    assert manifest["selected_dates"] == ["20260914", "20260915"]


def test_explicit_dates_count_larger_than_available_still_errors(
    tmp_path: Path,
) -> None:
    options = make_options(
        tmp_path / "explicit-short-range",
        dates=10,
        dates_explicit=True,
        explicit_dates=None,
        snapshot_only=True,
        use_default_explicit_date=False,
    )

    with pytest.raises(
        ValueError, match="요청한 날짜 수가 기간 내 거래일 수보다 큽니다"
    ):
        make_runner(
            options,
            calendar_dates=["20260914", "20260915"],
        ).run()


def test_masking_round_trip_restores_recommendation_code_and_prices() -> None:
    mapping = MaskingMap.create(["005930"], 123)
    recommendation = StockRecommendation(
        "005930",
        Action.BUY,
        0.8,
        Decimal("120"),
        Decimal("90"),
        "근거",
        original_invalidation_price=Decimal("95"),
        price_adjusted=True,
    )
    alias = mapping.masked_code("005930")
    masked = replace(
        recommendation,
        code=alias,
        target_price=mapping.scale("005930", recommendation.target_price),
        invalidation_price=mapping.scale("005930", recommendation.invalidation_price),
        original_invalidation_price=mapping.scale(
            "005930", recommendation.original_invalidation_price
        ),
    )

    restored = restore_recommendation(masked, mapping)
    assert restored.code == recommendation.code
    assert abs(restored.target_price - recommendation.target_price) < Decimal("0.0001")
    assert abs(
        restored.invalidation_price - recommendation.invalidation_price
    ) < Decimal("0.0001")
    assert abs(
        restored.original_invalidation_price
        - recommendation.original_invalidation_price
    ) < Decimal("0.0001")
    assert restored.price_adjusted is True


def test_mask_snapshot_removes_dates_names_news_and_rate_levels() -> None:
    mapping = MaskingMap.create(["005930"], 1)
    snapshot = ResearchSnapshot(
        created_at=datetime.fromisoformat("2026-09-15T15:40:00+09:00"),
        stocks=[
            StockSnapshot(
                "005930",
                "삼성전자",
                current_price=Decimal("100"),
                macd=Decimal("2"),
                macd_signal_value=Decimal("1"),
                news_headlines=["기사"],
                industry="전자",
            )
        ],
        market_context=MarketContext(
            exchange_rate=Decimal("1400"), korea_3y_yield=Decimal("3")
        ),
        missing=["005930:news(20260915 empty)"],
        as_of_trading_date="20260915",
    )

    masked = mask_snapshot(snapshot, mapping)
    assert masked.created_at.year == 2000
    assert masked.as_of_trading_date is None
    assert masked.stocks[0].name != "삼성전자"
    assert masked.stocks[0].news_headlines is None
    assert masked.stocks[0].industry is None
    assert masked.market_context.exchange_rate is None
    assert "20260915" not in str(masked.missing)
    assert masked.stocks[0].current_price != Decimal("100")
    assert masked.stocks[0].macd == mapping.scale("005930", Decimal("2"))
    assert masked.stocks[0].macd_signal_value == mapping.scale("005930", Decimal("1"))


class DemoGateway(FakeGateway):
    def __init__(self, decision_date: str) -> None:
        super().__init__()
        self.decision_at = datetime.fromisoformat(
            f"{decision_date[:4]}-{decision_date[4:6]}-{decision_date[6:]}T15:40:00+09:00"
        )
        self.leakage: dict[str, int] = {"news_articles": 0}


class FakeRecommendationModel(DecisionModel[ModelRecommendation]):
    def __init__(self, model_id: str, code: str, price: Decimal, fail: bool) -> None:
        self.model_id = model_id
        self.code = code
        self.price = price
        self.fail = fail
        self.calls = 0

    def decide(self, prompt: str) -> ModelRun[ModelRecommendation]:
        self.calls += 1
        raw = (
            f'{{"recommendations":[{{"code":"{self.code}",'
            f'"target_price":"{self.price}"}}]}}'
        )
        if self.fail:
            return ModelRun(
                self.model_id,
                "stdout",
                "stderr",
                1,
                0.25,
                response_text=raw,
                error="fake model failure",
            )
        item = StockRecommendation(
            self.code,
            Action.BUY,
            0.8,
            self.price,
            self.price * Decimal("0.75"),
            f"{self.model_id} 근거",
        )
        return ModelRun(
            self.model_id,
            "stdout",
            "",
            0,
            0.25,
            response_text=raw,
            decision=ModelRecommendation(self.model_id, [item], "요약", []),
        )


class MemoryLedger:
    def __init__(self) -> None:
        self.next_id = 1
        self.finished = []

    def start_recommendation(self, started_at: datetime) -> int:
        run_id = self.next_id
        self.next_id += 1
        return run_id

    def finish_recommendation(self, result, finished_at: datetime) -> None:
        self.finished.append(result)


class FakeStats:
    def __init__(self) -> None:
        self.counts = (0, 0)

    def __call__(self) -> tuple[int, int]:
        return self.counts


def _weekday_calendar(start: str, end: str) -> list[str]:
    current = datetime.strptime(start, "%Y%m%d").date()
    final = datetime.strptime(end, "%Y%m%d").date()
    dates = []
    while current <= final:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def make_options(
    output_dir: Path,
    *,
    masked: bool = False,
    snapshot_only: bool = False,
    requested_to: str | None = "20260915",
    explicit_dates: list[str] | None = None,
    dates: str | int = "all",
    dates_explicit: bool = True,
    seed: int = 20261103,
    resume: bool = True,
    use_default_explicit_date: bool = True,
) -> BacktestOptions:
    return BacktestOptions(
        start_date="20260901",
        requested_to=requested_to,
        dates=dates,
        dates_explicit=dates_explicit,
        explicit_dates=(
            ["20260915"]
            if explicit_dates is None and use_default_explicit_date
            else explicit_dates
        ),
        seed=seed,
        universe=["005930"],
        mask=masked,
        horizons=[5, 10, 20],
        snapshot_only=snapshot_only,
        workers=1,
        decision_time=time(15, 40),
        resume=resume,
        output_dir=output_dir,
        database=output_dir / "backtest.db",
        cache_dir=output_dir / "cache",
        consensus={"policy": "buy_one_side", "confidence_aggregation": "mean"},
        models={
            "claude": {"command": "claude", "model": "fake-claude", "effort": "high"},
            "codex": {"command": "codex", "model": "fake-codex", "effort": "high"},
        },
        git_sha="fake-sha",
    )


def make_runner(
    options: BacktestOptions,
    *,
    fail_model: str | None = None,
    recommendation_error: Callable[[], bool] | None = None,
    calendar_dates: list[str] | None = None,
    now: Callable[[], datetime] | None = None,
    calendar_calls: list[str] | None = None,
    model_calls: list[FakeRecommendationModel] | None = None,
    progress_events: list[tuple] | None = None,
    gateway_ranges: list[tuple[str, str]] | None = None,
) -> RunBacktest:
    store = FileBacktestResultStore(options.output_dir)
    ledger = MemoryLedger()
    stats = FakeStats()
    calendar_call_list = calendar_calls if calendar_calls is not None else []
    model_call_list = model_calls if model_calls is not None else []
    progress = progress_events if progress_events is not None else []
    ranges = gateway_ranges if gateway_ranges is not None else []
    dates = calendar_dates or ["20260914", "20260915", "20260916"]

    def calendar_provider(start: str, end: str) -> list[str]:
        calendar_call_list.append(end)
        return [date for date in dates if start <= date <= end]

    def gateway_factory(date, range_start, range_end, decision_time):
        ranges.append((range_start, range_end))
        return DemoGateway(date)

    def snapshot_factory(gateway, codes, clock, mapping):
        builder = SnapshotBuilder(gateway, codes, clock=clock)
        return MaskedSnapshotProvider(builder, mapping) if mapping else builder

    def recommendation_factory(gateway, codes, clock, mapping):
        if recommendation_error is not None and recommendation_error():
            raise RuntimeError("transient recommendation construction failure")
        actual_code = codes[0]
        response_code = mapping.masked_code(actual_code) if mapping else actual_code
        response_price = (
            mapping.scale(actual_code, Decimal("120")) if mapping else Decimal("120")
        )
        claude = FakeRecommendationModel(
            "claude", response_code, response_price, fail_model == "claude"
        )
        codex = FakeRecommendationModel(
            "codex", response_code, response_price, fail_model == "codex"
        )
        model_call_list.extend((claude, codex))
        service = RecommendationConsensusService(claude, codex)
        return RecommendStocks(
            snapshot_builder=snapshot_factory(gateway, codes, clock, mapping),
            consensus_service=service,
            universe=codes,
            ledger=ledger,
            clock=clock,
        )

    return RunBacktest(
        options=options,
        result_store=store,
        calendar_provider=calendar_provider,
        gateway_factory=gateway_factory,
        snapshot_factory=snapshot_factory,
        recommendation_factory=recommendation_factory,
        now=now or (lambda: datetime.fromisoformat("2026-11-03T16:00:00+09:00")),
        cache_stats=stats,
        progress=lambda *event: progress.append(event),
        warning=lambda _: None,
    )


@pytest.mark.parametrize("masked", [False, True])
def test_run_backtest_completed_writes_reusable_plain_and_masked_results(
    tmp_path: Path, masked: bool
) -> None:
    options = make_options(tmp_path / ("masked" if masked else "plain"), masked=masked)
    calls: list[FakeRecommendationModel] = []
    runner = make_runner(options, model_calls=calls)

    assert runner.run() == 0
    store = FileBacktestResultStore(options.output_dir)
    result = store.read_date_result("20260915")
    assert result is not None
    assert result["status"] == "completed"
    assert result["format_version"] == 1
    assert [item["model_id"] for item in result["models"]] == ["claude", "codex"]
    assert result["models"][0]["response_text"]
    assert result["models"][0]["decision"]["recommendations"]
    assert result["models"][0]["duration_seconds"] == 0.25
    assert result["consensus"]["recommendations"][0]["consensus_basis"] == "agreed"
    assert result["consensus"]["disagreements"] == []
    assert result["recommendations"][0]["code"] == "005930"
    assert result["recommendations"][0]["target_price"] == "120"
    assert result["recommendations"][0]["price_adjusted"] is False
    assert "original_invalidation_price" in result["recommendations"][0]
    assert result["recommendations"][0]["price_warnings"] == []
    assert (options.output_dir / "dates" / "20260915.prompt.txt").is_file()
    assert len(calls) == 2

    prompt = (options.output_dir / "dates" / "20260915.prompt.txt").read_text()
    if masked:
        alias = result["masking"]["codes"].popitem()[0]
        assert result["models"][0]["decision"]["recommendations"][0]["code"] == alias
        assert (
            result["models"][0]["restored_decision"]["recommendations"][0]["code"]
            == "005930"
        )
        assert result["consensus"]["recommendations"][0]["code"] == alias
        assert result["consensus"]["restored"]["recommendations"][0]["code"] == "005930"
        assert alias in prompt
        assert "005930" not in prompt
        assert "2026" not in prompt
        multiplier = Decimal(result["masking"]["price_multipliers"][alias])
        masked_price = Decimal(
            result["models"][0]["decision"]["recommendations"][0]["target_price"]
        )
        assert masked_price == Decimal("120") * multiplier
        assert Decimal(
            result["models"][0]["restored_decision"]["recommendations"][0][
                "target_price"
            ]
        ) == Decimal("120")
    else:
        assert "005930" in prompt
        assert "2026" in prompt
        assert "restored_decision" not in result["models"][0]


@pytest.mark.parametrize("masked", [False, True])
def test_run_backtest_model_failed_writes_both_runs_and_returns_two(
    tmp_path: Path, masked: bool
) -> None:
    options = make_options(
        tmp_path / ("model-failed-masked" if masked else "model-failed"), masked=masked
    )
    runner = make_runner(options, fail_model="claude")

    assert runner.run() == 2
    result = FileBacktestResultStore(options.output_dir).read_date_result("20260915")
    assert result is not None
    assert result["status"] == "model_failed"
    assert result["models"][0]["error"] == "fake model failure"
    assert result["models"][1]["decision"] is not None
    if masked:
        assert (
            result["models"][1]["restored_decision"]["recommendations"][0]["code"]
            == "005930"
        )
        assert result["masking"]["codes"]
    assert result["consensus"] is None
    assert (options.output_dir / "dates" / "20260915.prompt.txt").is_file()


def test_run_backtest_snapshot_only_never_calls_recommendation_factory(
    tmp_path: Path,
) -> None:
    options = make_options(tmp_path / "snapshot", snapshot_only=True)
    runner = make_runner(options)
    runner.recommendation_factory = lambda *args: pytest.fail(
        "snapshot-only must not create models or a ledger"
    )

    assert runner.run() == 0
    result = FileBacktestResultStore(options.output_dir).read_date_result("20260915")
    assert result is not None
    assert result["status"] == "snapshot_only"
    assert result["format_version"] == 1
    assert "models" not in result
    assert not options.database.exists()


def test_run_backtest_resume_skips_completed_date_without_recalling_models(
    tmp_path: Path,
) -> None:
    options = make_options(tmp_path / "resume")
    calls: list[FakeRecommendationModel] = []
    assert make_runner(options, model_calls=calls).run() == 0

    events: list[tuple] = []
    second = make_runner(options, model_calls=calls, progress_events=events)
    assert second.run() == 0
    assert len(calls) == 2
    assert events[0][0:2] == ("20260915", "completed")
    assert events[0][2] > 0
    assert events[0][4] is True


def test_run_backtest_resume_reexecutes_error_date(tmp_path: Path) -> None:
    options = make_options(tmp_path / "retry")
    error_once = [True]

    def fail_once() -> bool:
        if error_once[0]:
            error_once[0] = False
            return True
        return False

    assert make_runner(options, recommendation_error=fail_once).run() == 1
    previous = FileBacktestResultStore(options.output_dir).read_date_result("20260915")
    assert previous is not None and previous["status"] == "error"

    assert make_runner(options).run() == 0
    current = FileBacktestResultStore(options.output_dir).read_date_result("20260915")
    assert current is not None and current["status"] == "completed"


def test_run_backtest_resume_reexecutes_model_failed_date(tmp_path: Path) -> None:
    options = make_options(tmp_path / "model-retry")
    calls: list[FakeRecommendationModel] = []

    assert make_runner(options, fail_model="claude", model_calls=calls).run() == 2
    previous = FileBacktestResultStore(options.output_dir).read_date_result("20260915")
    assert previous is not None and previous["status"] == "model_failed"

    assert make_runner(options, model_calls=calls).run() == 0
    current = FileBacktestResultStore(options.output_dir).read_date_result("20260915")
    assert current is not None and current["status"] == "completed"
    assert len(calls) == 4


def test_run_backtest_resume_reexecutes_corrupt_date_result_with_warning(
    tmp_path: Path,
) -> None:
    options = make_options(tmp_path / "corrupt-resume")
    calls: list[FakeRecommendationModel] = []
    assert make_runner(options, model_calls=calls).run() == 0
    result_path = options.output_dir / "dates" / "20260915.json"
    result_path.write_text("{trunc", encoding="utf-8")

    warnings: list[str] = []
    rerun = make_runner(options, model_calls=calls)
    rerun.warning = warnings.append
    assert rerun.run() == 0

    assert len(calls) == 4
    assert len(warnings) == 1
    assert "20260915" in warnings[0]
    assert "재실행합니다" in warnings[0]
    result = FileBacktestResultStore(options.output_dir).read_date_result("20260915")
    assert result is not None and result["status"] == "completed"


def test_run_backtest_rejects_manifest_mismatch(tmp_path: Path) -> None:
    options = make_options(tmp_path / "manifest")
    assert make_runner(options).run() == 0

    changed = make_options(options.output_dir, masked=True)
    with pytest.raises(ValueError, match="설정이 다릅니다"):
        make_runner(changed).run()


def test_run_backtest_resume_ignores_workers_output_and_cache_paths(
    tmp_path: Path,
) -> None:
    options = make_options(tmp_path / "inert-resume-arguments")
    calls: list[FakeRecommendationModel] = []
    assert make_runner(options, model_calls=calls).run() == 0

    output_alias = Path(f"{options.output_dir}/../{options.output_dir.name}")
    changed = replace(
        options,
        workers=2,
        output_dir=output_alias,
        cache_dir=tmp_path / "different-cache",
    )
    events: list[tuple] = []
    assert make_runner(changed, model_calls=calls, progress_events=events).run() == 0
    assert len(calls) == 2
    assert events[0][4] is True


@pytest.mark.parametrize("requested_to", [None, "20261103"])
def test_run_backtest_persists_effective_to_reuses_it_and_excludes_intraday_today(
    tmp_path: Path, requested_to: str | None
) -> None:
    options = make_options(
        tmp_path / "effective-to",
        requested_to=requested_to,
        explicit_dates=None,
        dates=1,
        snapshot_only=True,
        use_default_explicit_date=False,
    )
    dates = _weekday_calendar("20260901", "20261104")

    def first_now() -> datetime:
        return datetime.fromisoformat("2026-11-03T10:00:00+09:00")

    calls: list[str] = []
    first = make_runner(
        options, calendar_dates=dates, now=first_now, calendar_calls=calls
    )

    assert first.run() == 0
    manifest = FileBacktestResultStore(options.output_dir).read_manifest()
    assert manifest is not None
    effective_to = manifest["effective_to"]
    assert manifest["arguments"]["effective_to"] == effective_to
    assert effective_to < "20261103"
    assert calls == ["20261102"]

    calls.clear()
    second = make_runner(
        options,
        calendar_dates=dates,
        now=lambda: datetime.fromisoformat("2026-11-04T16:00:00+09:00"),
        calendar_calls=calls,
    )
    assert second.run() == 0
    assert calls == [effective_to]
    new_manifest = FileBacktestResultStore(options.output_dir).read_manifest()
    assert new_manifest is not None
    assert new_manifest["effective_to"] == effective_to


class JsonRequester:
    base_url = "http://private-address.invalid"

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def request_json(self, method, path, json_body=None, timeout_seconds=None):
        self.calls += 1
        if self.fail:
            raise MarketDataUnavailable("failure")
        return {"ok": True}

    def close(self) -> None:
        pass


def test_response_cache_hits_without_inner_call_and_cache_files_have_no_server_address(
    tmp_path: Path,
) -> None:
    inner = JsonRequester()
    cache = ResponseCache(inner, tmp_path)
    assert cache.request_json("GET", "/market") == {"ok": True}
    assert cache.request_json("GET", "/market") == {"ok": True}
    assert inner.calls == 1
    assert cache.hits == 1
    assert cache.misses == 1
    contents = "".join(path.read_text() for path in tmp_path.iterdir())
    assert "private-address" not in contents
    assert all(path.suffix == ".json" for path in tmp_path.iterdir())


def test_response_cache_does_not_cache_errors(tmp_path: Path) -> None:
    inner = JsonRequester(fail=True)
    cache = ResponseCache(inner, tmp_path)
    for _ in range(2):
        with pytest.raises(MarketDataUnavailable):
            cache.request_json("GET", "/market")
    assert inner.calls == 2
    assert cache.misses == 2
    assert list(tmp_path.iterdir()) == []


def test_atomic_cache_write_preserves_target_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch
) -> None:
    from trading_agent.infrastructure import response_cache

    target = tmp_path / "response.json"
    target.write_text("old", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(response_cache.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_backtest_cli_without_subcommand_prints_help_and_returns_2(capsys) -> None:
    assert main(["backtest"]) == 2
    output = capsys.readouterr().out
    assert "usage: trading-agent backtest" in output
    assert "{run,score}" in output


def test_backtest_runtime_rejects_operating_database_before_gateway_creation(
    tmp_path: Path, monkeypatch
) -> None:
    from trading_agent.interface import bootstrap

    operating_database = tmp_path / "operating.sqlite"
    settings = SimpleNamespace(ledger=SimpleNamespace(db_path=operating_database))
    args = SimpleNamespace(
        out=tmp_path / "output", cache_dir=tmp_path / "cache", db=operating_database
    )

    def unexpected_gateway(*_args, **_kwargs):
        pytest.fail("gateway must not be created for the operating database")

    monkeypatch.setattr(bootstrap, "TradingApiMarketDataGateway", unexpected_gateway)
    with pytest.raises(ValueError, match="ledger.db_path"):
        bootstrap.create_backtest_runtime(settings, args)


def test_configured_horizon_reaches_run_recommend_and_backtest_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_agent.interface.bootstrap import (
        create_backtest_runtime,
        create_recommendation_runtime,
        create_runtime,
    )

    for name in (
        "TRADING_AGENT__RECOMMENDATION__HORIZON_DAYS",
        "TRADING_AGENT__RECOMMENDATION__UNIVERSE",
        "TRADING_AGENT__LEDGER__DB_PATH",
        "TRADING_AGENT__WATCHLIST",
    ):
        monkeypatch.delenv(name, raising=False)
    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        f"recommendation:\n  universe: ['005930']\n  horizon_days: [5, 15]\n"
        f"ledger:\n  db_path: {tmp_path / 'agent.sqlite'}\n",
        encoding="utf-8",
    )
    settings = load_settings(config_file)
    created_at = datetime(2026, 11, 4)
    market_snapshot = MarketSnapshot(
        created_at=created_at,
        account=AccountSnapshot(Decimal("100000"), []),
        stocks=[],
        market_context=MarketContext(),
        missing=[],
    )
    research_snapshot = ResearchSnapshot(
        created_at=created_at,
        stocks=[],
        market_context=MarketContext(),
        missing=[],
    )

    run_runtime = create_runtime(settings, db_path=tmp_path / "run.sqlite")
    try:
        run_prompt = run_runtime.cycle.prompt_builder(
            market_snapshot,
            RiskPolicy(max_position_weight=Decimal("0.2")),
        )
        assert "5~15거래일(약 1~3주)" in run_prompt
    finally:
        run_runtime.close()

    recommendation_runtime = create_recommendation_runtime(
        settings, db_path=tmp_path / "recommend.sqlite"
    )
    try:
        recommend_prompt = recommendation_runtime.use_case.prompt_builder(
            research_snapshot
        )
        assert "5~15거래일(약 1~3주)" in recommend_prompt
    finally:
        recommendation_runtime.close()

    args = build_parser().parse_args(
        [
            "backtest",
            "run",
            "--snapshot-only",
            "--out",
            str(tmp_path / "backtest-out"),
            "--db",
            str(tmp_path / "backtest.sqlite"),
            "--cache-dir",
            str(tmp_path / "backtest-cache"),
        ]
    )
    backtest_runtime = create_backtest_runtime(settings, args)
    try:
        recommendation = backtest_runtime.use_case.recommendation_factory(
            object(), ["005930"], lambda: created_at, None
        )
        backtest_prompt = recommendation.prompt_builder(research_snapshot)
        assert "5~15거래일(약 1~3주)" in backtest_prompt
    finally:
        backtest_runtime.close()


def test_backtest_cli_prints_runtime_error_detail_and_closes_runtime(
    monkeypatch, capsys
) -> None:
    from trading_agent.interface import cli

    class BrokenRuntime:
        closed = False

        def run(self) -> int:
            raise RuntimeError("날짜별 입력을 확인하세요")

        def close(self) -> None:
            self.closed = True

    runtime = BrokenRuntime()
    monkeypatch.setattr(cli, "load_settings", lambda _config: object())

    exit_code = main(
        ["backtest", "run"],
        backtest_runtime_factory=lambda _settings, _args: runtime,
    )

    assert exit_code == 1
    assert runtime.closed
    assert "RuntimeError: 날짜별 입력을 확인하세요" in capsys.readouterr().err


def test_backtest_cli_parses_repeated_and_comma_separated_dates() -> None:
    args = build_parser().parse_args(
        [
            "backtest",
            "run",
            "--date",
            "20260915",
            "--date",
            "20260916,20260917",
            "--universe",
            "005930,000660",
        ]
    )
    assert args.date == ["20260915", "20260916,20260917"]
    assert args.universe == ["005930", "000660"]
    assert args.workers == 1
