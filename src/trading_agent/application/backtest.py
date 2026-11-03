"""백테스트 날짜·마스킹·결과 직렬화 규칙."""

import random
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from trading_agent.application.ports import (
    DailyBar,
    IndexPoint,
    IndicatorPoint,
    MarketIndex,
    NewsItem,
    StockQuote,
    TreasuryMaturity,
)
from trading_agent.application.records import RecommendationRunResult
from trading_agent.application.snapshot_builder import LOOKBACK_CALENDAR_DAYS
from trading_agent.domain.consensus import Disagreement
from trading_agent.domain.models import AccountSnapshot, ResearchSnapshot
from trading_agent.domain.recommendation import (
    RecommendationConsensusResult,
    StockRecommendation,
)

BACKTEST_FORMAT_VERSION = 1
MEMORY_LEAK_CUTOFF = "20260701"
BACKTEST_MANIFEST_FILENAME = "manifest.json"
BACKTEST_DATE_DIRECTORY = "dates"
BACKTEST_DATE_RESULT_TEMPLATE = "{date}.json"
BACKTEST_PROMPT_TEMPLATE = "{date}.prompt.txt"
STATUS_SNAPSHOT_ONLY = "snapshot_only"
STATUS_ERROR = "error"
STATUS_COMPLETED = "completed"
STATUS_MODEL_FAILED = "model_failed"
STATUS_DATA_FAILED = "data_failed"
RESUMABLE_STATUSES = frozenset((STATUS_SNAPSHOT_ONLY, STATUS_COMPLETED))
EXIT_CODE_BY_STATUS = {
    STATUS_SNAPSHOT_ONLY: 0,
    STATUS_COMPLETED: 0,
    "skipped": 0,
    STATUS_MODEL_FAILED: 2,
    STATUS_DATA_FAILED: 1,
    "error": 1,
}
KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


class BacktestGateway(Protocol):
    """PIT gateway가 유스케이스에 추가로 제공하는 판단 시각과 누수 집계."""

    decision_at: datetime
    leakage: Mapping[str, int]

    def get_account(self) -> AccountSnapshot: ...

    def get_stock_quote(self, code: str) -> StockQuote: ...

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]: ...

    def get_moving_average(
        self, code: str, start_date: str, end_date: str, window: int
    ) -> list[IndicatorPoint]: ...

    def get_rsi(
        self, code: str, start_date: str, end_date: str, window: int = 14
    ) -> list[IndicatorPoint]: ...

    def get_rsi_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        rsi_window: int = 14,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]: ...

    def get_macd(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
    ) -> list[IndicatorPoint]: ...

    def get_macd_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]: ...

    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]: ...

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]: ...

    def get_fear_and_greed_index(self) -> IndexPoint: ...

    def get_treasury_yield(
        self, maturity: TreasuryMaturity, start_date: str, end_date: str
    ) -> list[IndexPoint]: ...


class SnapshotProvider(Protocol):
    """추천과 snapshot-only 실행이 사용하는 스냅샷 팩토리 결과."""

    def build_research(self, codes: Sequence[str]) -> ResearchSnapshot: ...


class RecommendationRunner(Protocol):
    """기존 RecommendStocks 실행 인터페이스."""

    def run(self) -> RecommendationRunResult: ...


class BacktestResultStore(Protocol):
    """manifest, 날짜 결과, 프롬프트 저장 및 재개 조회 포트."""

    def read_manifest(self) -> Mapping[str, Any] | None: ...

    def read_date_result(self, date: str) -> Mapping[str, Any] | None: ...

    def write_manifest(self, value: Mapping[str, Any]) -> None: ...

    def write_date_result(self, date: str, value: Mapping[str, Any]) -> None: ...

    def write_prompt(self, date: str, value: str) -> None: ...


@dataclass(frozen=True, slots=True)
class BacktestOptions:
    """CLI에서 변환한 백테스트 입력과 산출 경로 메타데이터."""

    start_date: str
    requested_to: str | None
    dates: str | int
    dates_explicit: bool
    explicit_dates: list[str] | None
    seed: int
    universe: list[str]
    mask: bool
    horizons: list[int]
    snapshot_only: bool
    workers: int
    decision_time: time
    resume: bool
    output_dir: Path
    database: Path
    cache_dir: Path
    consensus: Mapping[str, str]
    models: Mapping[str, Any]
    git_sha: str | None = None


def choose_dates(
    calendar_dates: list[str],
    start_date: str,
    end_date: str,
    date_count: int | str,
    seed: int,
    explicit_dates: list[str] | None = None,
) -> list[str]:
    """거래일 구간에서 명시일 또는 seed 재현 가능한 날짜를 고른다."""

    available = sorted(
        date for date in set(calendar_dates) if start_date <= date <= end_date
    )
    if explicit_dates:
        requested = sorted(set(explicit_dates))
        outside = [date for date in requested if date < start_date or date > end_date]
        if outside:
            raise ValueError(
                "--date가 --from/--to 구간 밖입니다: " + ", ".join(outside)
            )
        unknown = [date for date in requested if date not in calendar_dates]
        if unknown:
            raise ValueError("거래일 목록에 없는 날짜: " + ", ".join(unknown))
        return requested
    if date_count == "all":
        return available
    count = int(date_count)
    if count <= 0:
        raise ValueError("--dates는 all 또는 양의 정수여야 합니다.")
    if count > len(available):
        raise ValueError("요청한 날짜 수가 기간 내 거래일 수보다 큽니다.")
    return sorted(random.Random(seed).sample(available, count))


def last_safe_date(calendar_dates: list[str], horizon: int) -> str:
    """미래 horizon 거래일 가격이 존재하는 마지막 거래일을 고른다."""

    dates = sorted(set(calendar_dates))
    if horizon <= 0:
        raise ValueError("horizon은 양수여야 합니다.")
    if len(dates) <= horizon:
        raise ValueError("미래 horizon 거래일이 확보된 날짜가 없습니다.")
    return dates[-horizon - 1]


@dataclass(frozen=True, slots=True)
class MaskingMap:
    """종목 코드와 가격의 재현 가능한 가역 마스킹 정보."""

    aliases: dict[str, str]
    multipliers: dict[str, Decimal]

    @classmethod
    def create(cls, codes: list[str], seed: int) -> "MaskingMap":
        rng = random.Random(seed)
        used = set(codes)
        aliases: dict[str, str] = {}
        multipliers: dict[str, Decimal] = {}
        for code in dict.fromkeys(codes):
            while True:
                alias = f"{rng.randrange(1, 1_000_000):06d}"
                if alias not in used:
                    used.add(alias)
                    break
            aliases[code] = alias
            multipliers[code] = Decimal(str(0.5 + rng.random() * 1.5)).quantize(
                Decimal("0.000001")
            )
        return cls(aliases, multipliers)

    def masked_code(self, code: str) -> str:
        return self.aliases.get(code, code)

    def original_code(self, alias: str) -> str:
        return next(
            (code for code, value in self.aliases.items() if value == alias), alias
        )

    def scale(self, code: str, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        multiplier = self.multipliers.get(self.original_code(code))
        return value if multiplier is None else value * multiplier

    def unscale(self, code: str, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        original = self.original_code(code)
        multiplier = self.multipliers.get(original)
        return value if multiplier is None else value / multiplier

    def as_json(self) -> dict[str, Any]:
        """결과 파일에 필요한 역변환표를 반환한다."""

        return {
            "codes": {alias: code for code, alias in self.aliases.items()},
            "price_multipliers": {
                self.aliases[code]: str(multiplier)
                for code, multiplier in self.multipliers.items()
            },
        }


SnapshotFactory = Callable[
    [BacktestGateway, Sequence[str], Callable[[], datetime], MaskingMap | None],
    SnapshotProvider,
]
RecommendationFactory = Callable[
    [BacktestGateway, Sequence[str], Callable[[], datetime], MaskingMap | None],
    RecommendationRunner,
]
GatewayFactory = Callable[[str, str, str, time], BacktestGateway]
CalendarProvider = Callable[[str, str], Sequence[str]]
CacheStatsProvider = Callable[[], tuple[int, int]]
ProgressCallback = Callable[[str, str, int, int, bool, str | None], None]


def mask_snapshot(snapshot: ResearchSnapshot, mapping: MaskingMap) -> ResearchSnapshot:
    """프롬프트용 스냅샷에서 식별자·시각·뉴스·금리 수준을 감춘다."""

    stocks = [
        replace(
            stock,
            code=mapping.masked_code(stock.code),
            name="마스킹 종목",
            current_price=mapping.scale(stock.code, stock.current_price),
            macd=mapping.scale(stock.code, stock.macd),
            macd_signal_value=mapping.scale(stock.code, stock.macd_signal_value),
            news_headlines=None,
        )
        for stock in snapshot.stocks
    ]
    context = replace(
        snapshot.market_context,
        exchange_rate=None,
        korea_3y_yield=None,
        us_10y_yield=None,
    )
    missing = []
    for item in snapshot.missing:
        for original, alias in mapping.aliases.items():
            item = item.replace(original, alias)
        missing.append(re.sub(r"\b\d{8}\b", "날짜", item))
    return replace(
        snapshot,
        created_at=datetime(2000, 1, 1, tzinfo=snapshot.created_at.tzinfo),
        stocks=stocks,
        market_context=context,
        missing=missing,
        as_of_trading_date=None,
        market_session="closed",
    )


class MaskedSnapshotProvider:
    """기존 스냅샷 제공자의 반환값을 모델 입력 전에 마스킹한다."""

    def __init__(self, inner: SnapshotProvider, mapping: MaskingMap) -> None:
        self.inner = inner
        self.mapping = mapping

    def build_research(self, codes: Sequence[str]) -> ResearchSnapshot:
        return mask_snapshot(self.inner.build_research(codes), self.mapping)


def restore_recommendation(
    recommendation: StockRecommendation, mapping: MaskingMap
) -> StockRecommendation:
    """마스킹 추천의 종목 코드와 가격을 원래 값으로 복구한다."""

    return replace(
        recommendation,
        code=mapping.original_code(recommendation.code),
        target_price=mapping.unscale(recommendation.code, recommendation.target_price),
        invalidation_price=mapping.unscale(
            recommendation.code, recommendation.invalidation_price
        ),
        original_invalidation_price=mapping.unscale(
            recommendation.code, recommendation.original_invalidation_price
        ),
    )


def json_value(value: Any) -> Any:
    """도메인 결과와 표준 값을 JSON 직렬화 가능한 구조로 바꾼다."""

    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return {
            name: json_value(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [json_value(item) for item in items]
    return value


class RunBacktest:
    """날짜별 PIT 스냅샷·추천을 실행하고 안정된 결과 포맷으로 저장한다."""

    def __init__(
        self,
        options: BacktestOptions,
        result_store: BacktestResultStore,
        calendar_provider: CalendarProvider,
        gateway_factory: GatewayFactory,
        snapshot_factory: SnapshotFactory,
        recommendation_factory: RecommendationFactory,
        now: Callable[[], datetime],
        cache_stats: CacheStatsProvider,
        progress: ProgressCallback,
        warning: Callable[[str], None],
    ) -> None:
        self.options = options
        self.result_store = result_store
        self.calendar_provider = calendar_provider
        self.gateway_factory = gateway_factory
        self.snapshot_factory = snapshot_factory
        self.recommendation_factory = recommendation_factory
        self.now = now
        self.cache_stats = cache_stats
        self.progress = progress
        self.warning = warning

    def run(self) -> int:
        """유효 거래일을 고르고, 저장소 포트를 통해 날짜 결과를 남긴다."""

        options = self.options
        started_at = self.now()
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("now 포트는 timezone-aware datetime을 반환해야 합니다.")
        local_now = started_at.astimezone(KOREA_TIMEZONE)
        today = local_now.strftime("%Y%m%d")
        market_calendar_end = today
        if local_now.time() < time(15, 30):
            market_calendar_end = (local_now.date() - timedelta(days=1)).strftime(
                "%Y%m%d"
            )

        existing_manifest = (
            self.result_store.read_manifest() if options.resume else None
        )
        saved_effective_to = (
            existing_manifest.get("effective_to")
            if existing_manifest is not None
            else None
        )
        if saved_effective_to is not None:
            calendar_end = min(saved_effective_to, market_calendar_end)
        elif options.requested_to is not None:
            calendar_end = min(options.requested_to, market_calendar_end)
        else:
            calendar_end = market_calendar_end

        calendar_dates = sorted(
            set(self.calendar_provider(options.start_date, calendar_end))
        )
        if not calendar_dates:
            raise ValueError("KOSPI 거래일 목록이 비어 있습니다.")
        if saved_effective_to is not None:
            effective_to = saved_effective_to
        elif options.requested_to is not None:
            effective_to = min(options.requested_to, market_calendar_end)
        else:
            effective_to = last_safe_date(calendar_dates, max(options.horizons))
        selected_dates = choose_dates(
            calendar_dates,
            options.start_date,
            effective_to,
            (
                min(
                    10,
                    sum(
                        options.start_date <= date <= effective_to
                        for date in set(calendar_dates)
                    ),
                )
                if not options.dates_explicit
                else options.dates
            ),
            options.seed,
            options.explicit_dates,
        )
        if not selected_dates:
            raise ValueError("실행할 거래일이 없습니다.")
        if not options.universe:
            raise ValueError("백테스트 universe가 비어 있습니다.")
        if any(len(code) != 6 or not code.isdigit() for code in options.universe):
            raise ValueError("universe는 6자리 종목 코드 목록이어야 합니다.")
        if options.start_date < MEMORY_LEAK_CUTOFF:
            self.warning(
                "경고: 2026-07-01 이전 날짜는 모델 기억 누수 가능성이 있습니다."
            )

        mapping = (
            MaskingMap.create(options.universe, options.seed) if options.mask else None
        )
        range_start = (
            datetime.strptime(min(selected_dates), "%Y%m%d")
            - timedelta(days=LOOKBACK_CALENDAR_DAYS + 14)
        ).strftime("%Y%m%d")
        range_end = max(selected_dates)
        self._validate_manifest(existing_manifest, selected_dates, effective_to)
        manifest = self._manifest(started_at, selected_dates, effective_to, options)
        self.result_store.write_manifest(manifest)

        def run_date(decision_date: str) -> int:
            if options.resume and existing_manifest is not None:
                try:
                    previous = self.result_store.read_date_result(decision_date)
                except (OSError, ValueError) as error:
                    self.warning(
                        f"경고: {decision_date} 날짜 결과를 읽지 못해 재실행합니다 "
                        f"({type(error).__name__})."
                    )
                    previous = None
                if (
                    previous is not None
                    and previous.get("status") in RESUMABLE_STATUSES
                ):
                    self.progress(
                        decision_date,
                        str(previous.get("status")),
                        len(previous.get("missing", [])),
                        sum(previous.get("leakage", {}).values()),
                        True,
                        None,
                    )
                    return EXIT_CODE_BY_STATUS.get(str(previous.get("status")), 1)

            cache_before = self.cache_stats()
            gateway: BacktestGateway | None = None
            prompt: str | None = None
            try:
                gateway = self.gateway_factory(
                    decision_date, range_start, range_end, options.decision_time
                )

                def clock(gateway: BacktestGateway = gateway) -> datetime:
                    return gateway.decision_at

                if options.snapshot_only:
                    snapshot_provider = self.snapshot_factory(
                        gateway, options.universe, clock, mapping
                    )
                    snapshot = snapshot_provider.build_research(options.universe)
                    result_data = {
                        "format_version": BACKTEST_FORMAT_VERSION,
                        "date": decision_date,
                        "status": STATUS_SNAPSHOT_ONLY,
                        "snapshot": json_value(snapshot),
                        "missing": list(snapshot.missing),
                    }
                    exit_code = 0
                else:
                    recommendation = self.recommendation_factory(
                        gateway, options.universe, clock, mapping
                    ).run()
                    prompt = recommendation.prompt
                    result_data = recommendation_result_json(recommendation, mapping)
                    result_data.update(
                        {
                            "format_version": BACKTEST_FORMAT_VERSION,
                            "date": decision_date,
                        }
                    )
                    exit_code = EXIT_CODE_BY_STATUS.get(recommendation.status.value, 1)

                leakage = dict(gateway.leakage)
                cache_after = self.cache_stats()
                result_data.update(
                    {
                        "leakage": leakage,
                        "cache": {
                            "hits": cache_after[0] - cache_before[0],
                            "misses": cache_after[1] - cache_before[1],
                        },
                        "mode": "masked" if mapping is not None else "plain",
                    }
                )
                if mapping is not None:
                    result_data["masking"] = mapping.as_json()
                self.result_store.write_date_result(decision_date, result_data)
                if prompt is not None:
                    self.result_store.write_prompt(decision_date, prompt)
                self.progress(
                    decision_date,
                    str(result_data["status"]),
                    len(result_data.get("missing", [])),
                    sum(leakage.values()),
                    False,
                    result_data.get("error"),
                )
                return exit_code
            except Exception as error:
                leakage = dict(gateway.leakage) if gateway is not None else {}
                cache_after = self.cache_stats()
                result_data = {
                    "format_version": BACKTEST_FORMAT_VERSION,
                    "date": decision_date,
                    "status": STATUS_ERROR,
                    "error": f"{type(error).__name__}: {error}",
                    "missing": [],
                    "leakage": leakage,
                    "cache": {
                        "hits": cache_after[0] - cache_before[0],
                        "misses": cache_after[1] - cache_before[1],
                    },
                    "mode": "masked" if mapping is not None else "plain",
                }
                if mapping is not None:
                    result_data["masking"] = mapping.as_json()
                self.result_store.write_date_result(decision_date, result_data)
                self.progress(
                    decision_date,
                    STATUS_ERROR,
                    0,
                    sum(leakage.values()),
                    False,
                    result_data["error"],
                )
                return 1

        if options.workers == 1:
            exit_code = max(run_date(date) for date in selected_dates)
        else:
            with ThreadPoolExecutor(
                max_workers=options.workers, thread_name_prefix="backtest-date"
            ) as pool:
                exit_code = max(pool.map(run_date, selected_dates))
        manifest["finished_at"] = self.now().isoformat()
        self.result_store.write_manifest(manifest)
        return exit_code

    def _validate_manifest(
        self,
        existing: Mapping[str, Any] | None,
        selected_dates: list[str],
        effective_to: str,
    ) -> None:
        if not self.options.resume or existing is None:
            return
        expected = self._manifest(None, selected_dates, effective_to, self.options)
        ignored_arguments = {"workers", "out", "cache_dir"}
        existing_arguments = existing.get("arguments")
        expected_arguments = expected["arguments"]
        arguments_match = isinstance(existing_arguments, Mapping) and {
            key: value
            for key, value in existing_arguments.items()
            if key not in ignored_arguments
        } == {
            key: value
            for key, value in expected_arguments.items()
            if key not in ignored_arguments
        }
        if not arguments_match or any(
            existing.get(key) != expected.get(key)
            for key in (
                "format_version",
                "mode",
                "snapshot_only",
                "selected_dates",
                "effective_to",
                "consensus",
                "models",
            )
        ):
            raise ValueError("기존 백테스트와 설정이 다릅니다.")

    @staticmethod
    def _manifest(
        started_at: datetime | None,
        selected_dates: list[str],
        effective_to: str,
        options: BacktestOptions,
    ) -> dict[str, Any]:
        """Step 3가 재사용할 manifest 스키마를 조립한다."""

        return {
            "format_version": BACKTEST_FORMAT_VERSION,
            "mode": "masked" if options.mask else "plain",
            "snapshot_only": options.snapshot_only,
            "arguments": {
                "from": options.start_date,
                "to": options.requested_to,
                "effective_to": effective_to,
                "dates": options.dates,
                "explicit_dates": options.explicit_dates,
                "seed": options.seed,
                "universe": list(options.universe),
                "horizons": list(options.horizons),
                "mask": options.mask,
                "workers": options.workers,
                "decision_time": options.decision_time.strftime("%H:%M"),
                "out": str(options.output_dir),
                "db": str(options.database),
                "cache_dir": str(options.cache_dir),
            },
            "selected_dates": selected_dates,
            "effective_to": effective_to,
            "consensus": dict(options.consensus),
            "models": json_value(options.models),
            "git_sha": options.git_sha,
            "started_at": started_at.isoformat() if started_at else None,
            "finished_at": None,
            "memory_leak_warning": options.start_date < MEMORY_LEAK_CUTOFF,
        }


def recommendation_result_json(
    result: RecommendationRunResult, mapping: MaskingMap | None
) -> dict[str, Any]:
    """추천 결과를 저장 형식으로 바꾸고 마스킹 추천의 복구값도 기록한다."""

    outcome = result.outcome
    models = []
    if outcome is not None:
        for model_run in (outcome.claude_run, outcome.codex_run):
            model_data = json_value(model_run)
            if mapping and model_run.decision is not None:
                restored = replace(
                    model_run.decision,
                    recommendations=[
                        restore_recommendation(item, mapping)
                        for item in model_run.decision.recommendations
                    ],
                )
                model_data["restored_decision"] = json_value(restored)
            models.append(model_data)

    consensus = outcome.consensus if outcome is not None else None
    consensus_data = json_value(consensus)
    if mapping and consensus is not None:
        consensus_data["restored"] = json_value(_restore_consensus(consensus, mapping))
    recommendations = list(result.recommendations or [])
    if mapping:
        recommendations = [
            restore_recommendation(item, mapping) for item in recommendations
        ]
    snapshot = result.snapshot
    result_data: dict[str, Any] = {
        "status": result.status.value,
        "error": result.error,
        "snapshot": json_value(snapshot),
        "missing": list(snapshot.missing) if snapshot is not None else [],
        "models": models,
        "consensus": consensus_data,
        "recommendations": json_value(recommendations),
    }
    if mapping is not None:
        result_data["masking"] = mapping.as_json()
    return result_data


def _restore_consensus(
    consensus: RecommendationConsensusResult, mapping: MaskingMap
) -> RecommendationConsensusResult:
    disagreements: list[Disagreement] = [
        replace(item, code=mapping.original_code(item.code))
        for item in consensus.disagreements
    ]
    return replace(
        consensus,
        recommendations=[
            restore_recommendation(item, mapping) for item in consensus.recommendations
        ],
        disagreements=disagreements,
    )
