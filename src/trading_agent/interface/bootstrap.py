"""CLI 인자와 설정에서 application 유스케이스·외부 어댑터를 조립한다."""

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any
from zoneinfo import ZoneInfo

from trading_agent.application.backtest import (
    BacktestGateway,
    BacktestOptions,
    MaskedSnapshotProvider,
    RunBacktest,
)
from trading_agent.application.backtest_score import (
    BacktestScoreOptions,
    ScoreBacktest,
)
from trading_agent.application.consensus_service import (
    ConsensusDecisionService,
    RecommendationConsensusService,
)
from trading_agent.application.model_tasks import DECISION_TASK, RECOMMENDATION_TASK
from trading_agent.application.ports import MarketIndex
from trading_agent.application.recommend_stocks import RecommendStocks
from trading_agent.application.records import (
    DecisionCycleResult,
    RecommendationRunResult,
)
from trading_agent.application.run_decision_cycle import RunDecisionCycle
from trading_agent.application.snapshot_builder import SnapshotBuilder
from trading_agent.domain.models import ConfidenceAggregation, ConsensusPolicyName
from trading_agent.domain.risk import RiskPolicy
from trading_agent.infrastructure.backtest_result_store import FileBacktestResultStore
from trading_agent.infrastructure.command_runner import SubprocessCommandRunner
from trading_agent.infrastructure.config import AgentSettings
from trading_agent.infrastructure.decision_models import (
    ClaudeCliDecisionModel,
    CodexCliDecisionModel,
)
from trading_agent.infrastructure.point_in_time_gateway import (
    PointInTimeGateway,
    SharedDailyCache,
)
from trading_agent.infrastructure.response_cache import ResponseCache, atomic_write_text
from trading_agent.infrastructure.sqlite_ledger import SQLiteDecisionLedger
from trading_agent.infrastructure.trading_api import TradingApiMarketDataGateway

KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


def korea_now() -> datetime:
    """현재 timezone-aware 한국 시각을 반환한다."""

    return datetime.now(KOREA_TIMEZONE)


@dataclass(slots=True)
class TradingAgentRuntime:
    """한 번의 실행과 HTTP 어댑터 정리를 제공한다."""

    cycle: RunDecisionCycle
    gateway: TradingApiMarketDataGateway

    def run(self) -> DecisionCycleResult:
        """조립된 사이클을 실행한다."""

        return self.cycle.run()

    def close(self) -> None:
        """trading-api HTTP 연결을 닫는다."""

        self.gateway.close()


def create_runtime(
    settings: AgentSettings,
    db_path: str | Path | None = None,
    force: bool = False,
) -> TradingAgentRuntime:
    """설정을 적용한 네트워크·LLM 어댑터와 사이클을 조립한다."""

    clock = korea_now
    gateway = TradingApiMarketDataGateway(settings.trading_api)
    runner = SubprocessCommandRunner()
    consensus_service = ConsensusDecisionService(
        ClaudeCliDecisionModel(settings.llm.claude, runner, DECISION_TASK),
        CodexCliDecisionModel(settings.llm.codex, runner, DECISION_TASK),
        policy=settings.consensus.policy,
        confidence_aggregation=settings.consensus.confidence_aggregation,
    )
    risk_policy = RiskPolicy(
        max_position_weight=Decimal(str(settings.risk.max_position_weight)),
        max_order_amount=(
            Decimal(str(settings.risk.max_order_amount))
            if settings.risk.max_order_amount is not None
            else None
        ),
        min_confidence=settings.risk.min_confidence,
        allowed_codes=set(settings.watchlist),
        max_proposals_per_cycle=settings.risk.max_proposals_per_cycle,
        max_order_ratio=Decimal(str(settings.risk.max_order_ratio)),
        stop_loss_percent=(
            Decimal(str(settings.risk.stop_loss_percent))
            if settings.risk.stop_loss_percent is not None
            else None
        ),
        take_profit_percent=(
            Decimal(str(settings.risk.take_profit_percent))
            if settings.risk.take_profit_percent is not None
            else None
        ),
        take_profit_sell_fraction=Decimal(str(settings.risk.take_profit_sell_fraction)),
    )
    cycle = RunDecisionCycle(
        snapshot_builder=SnapshotBuilder(gateway, settings.watchlist, clock=clock),
        consensus_service=consensus_service,
        risk_policy=risk_policy,
        watchlist=settings.watchlist,
        ledger=SQLiteDecisionLedger(
            db_path or settings.ledger.db_path,
            stale_run_after=timedelta(seconds=settings.ledger.stale_run_after_seconds),
        ),
        clock=clock,
        force=force,
    )
    return TradingAgentRuntime(cycle=cycle, gateway=gateway)


@dataclass(slots=True)
class RecommendationRuntime:
    """한 번의 종목 추천 실행과 HTTP 어댑터 정리를 제공한다."""

    use_case: RecommendStocks
    gateway: TradingApiMarketDataGateway

    def run(self) -> RecommendationRunResult:
        """조립된 추천 유스케이스를 실행한다."""

        return self.use_case.run()

    def close(self) -> None:
        """trading-api HTTP 연결을 닫는다."""

        self.gateway.close()


@dataclass(slots=True)
class BacktestRuntime:
    """application 백테스트 유스케이스와 HTTP 어댑터의 수명주기."""

    use_case: RunBacktest
    cache: ResponseCache

    def run(self) -> int:
        """조립된 application 유스케이스를 실행한다."""

        return self.use_case.run()

    def close(self) -> None:
        """응답 캐시와 그 안의 HTTP 클라이언트를 닫는다."""

        self.cache.close()


@dataclass(slots=True)
class BacktestScoreRuntime:
    """채점 유스케이스 실행, 보고서 저장, HTTP 어댑터 정리를 제공한다."""

    use_case: ScoreBacktest
    gateway: TradingApiMarketDataGateway
    output_json: Path
    output_markdown: Path

    def run(self) -> None:
        """채점 결과를 JSON·Markdown으로 저장하고 핵심 지표를 출력한다."""

        scored = self.use_case.run()
        atomic_write_text(
            self.output_json,
            json.dumps(scored.data, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
        )
        atomic_write_text(self.output_markdown, scored.markdown)
        counts = scored.data["counts"]
        print(
            "백테스트 채점 완료: "
            f"{counts['completed_dates']}일, policy={scored.data['policy']}/"
            f"{scored.data['confidence_aggregation']}"
        )
        for horizon in scored.data["horizons"]:
            metric = scored.data["sources"]["consensus"]["horizons"][str(horizon)]
            print(
                f"합의 {horizon}일: buy {metric['buy_n']}건, "
                f"buy xs 평균 {metric['buy_mean_xs']}, "
                f"sell xs 평균 {metric['sell_mean_xs']}"
            )
        print(f"보고서: {self.output_json} / {self.output_markdown}")

    def close(self) -> None:
        """응답 캐시와 trading-api HTTP 연결을 닫는다."""

        self.gateway.close()


def create_backtest_runtime(settings: AgentSettings, args: Any) -> BacktestRuntime:
    """CLI 값을 application 옵션으로 바꾸고 백테스트 포트 구현을 조립한다."""

    output_dir = Path(
        args.out or Path("data/backtest") / korea_now().strftime("%Y%m%dT%H%M%S%f")
    )
    cache_dir = Path(args.cache_dir or "data/backtest/cache")
    database = (
        Path(args.db).expanduser().resolve()
        if args.db is not None
        else (output_dir / "backtest.db").resolve()
    )
    configured_db = settings.ledger.db_path.expanduser().resolve()
    if database == configured_db:
        raise ValueError("--db는 설정의 ledger.db_path와 같은 파일일 수 없습니다.")

    universe = list(
        args.universe if args.universe is not None else settings.recommendation_universe
    )
    options = BacktestOptions(
        start_date=args.start_date,
        requested_to=args.end_date,
        dates=_parse_dates_count(args.dates),
        explicit_dates=_explicit_dates(args.date),
        seed=args.seed,
        universe=universe,
        mask=args.mask,
        horizons=_parse_positive_list(args.horizons),
        snapshot_only=args.snapshot_only,
        workers=args.workers,
        decision_time=args.decision_time,
        resume=args.resume,
        output_dir=output_dir,
        database=database,
        cache_dir=cache_dir,
        consensus={
            "policy": settings.consensus.policy.value,
            "confidence_aggregation": settings.consensus.confidence_aggregation.value,
        },
        models={
            "claude": {
                "command": settings.llm.claude.command,
                "model": settings.llm.claude.model,
                "effort": settings.llm.claude.effort,
            },
            "codex": {
                "command": settings.llm.codex.command,
                "model": settings.llm.codex.model,
                "effort": settings.llm.codex.reasoning_effort,
            },
        },
        git_sha=_git_sha(),
    )

    gateway = TradingApiMarketDataGateway(settings.trading_api)
    cache = ResponseCache(gateway.client, cache_dir)
    gateway.client = cache  # type: ignore[assignment]
    result_store = FileBacktestResultStore(output_dir)
    daily_cache = SharedDailyCache()
    ledger_lock = Lock()
    shared_ledger: SQLiteDecisionLedger | None = None
    runner = SubprocessCommandRunner()

    def calendar_provider(start_date: str, end_date: str) -> list[str]:
        points = gateway.get_market_index(MarketIndex.KOSPI, start_date, end_date)
        return [point.date for point in points]

    def gateway_factory(
        decision_date: str, range_start: str, range_end: str, decision_time: Any
    ) -> BacktestGateway:
        return PointInTimeGateway(
            gateway,
            decision_date,
            range_start,
            range_end,
            decision_time=decision_time,
            daily_cache=daily_cache,
        )

    def snapshot_factory(
        point_gateway: BacktestGateway,
        codes: list[str] | tuple[str, ...],
        clock: Any,
        mapping: Any,
    ) -> Any:
        builder = SnapshotBuilder(point_gateway, codes, clock=clock)
        return MaskedSnapshotProvider(builder, mapping) if mapping else builder

    def recommendation_factory(
        point_gateway: BacktestGateway,
        codes: list[str] | tuple[str, ...],
        clock: Any,
        mapping: Any,
    ) -> RecommendStocks:
        nonlocal shared_ledger
        snapshot_builder = snapshot_factory(point_gateway, codes, clock, mapping)
        with ledger_lock:
            if shared_ledger is None:
                shared_ledger = SQLiteDecisionLedger(
                    database,
                    stale_run_after=timedelta(
                        seconds=settings.ledger.stale_run_after_seconds
                    ),
                )
        service = RecommendationConsensusService(
            ClaudeCliDecisionModel(settings.llm.claude, runner, RECOMMENDATION_TASK),
            CodexCliDecisionModel(settings.llm.codex, runner, RECOMMENDATION_TASK),
            policy=settings.consensus.policy,
            confidence_aggregation=settings.consensus.confidence_aggregation,
        )
        assert shared_ledger is not None
        return RecommendStocks(
            snapshot_builder=snapshot_builder,  # type: ignore[arg-type]
            consensus_service=service,
            universe=codes,
            ledger=shared_ledger,
            clock=clock,
        )

    def progress(
        date: str,
        status: str,
        missing: int,
        leakage: int,
        skipped: bool,
        error: str | None,
    ) -> None:
        if skipped:
            print(f"{date}: resume skip")
        else:
            suffix = f" 오류: {error}" if error else ""
            print(f"{date}: {status} 누락 {missing}건, 누수 차단 {leakage}건{suffix}")

    def warning(message: str) -> None:
        print(message, file=sys.stderr)

    use_case = RunBacktest(
        options=options,
        result_store=result_store,
        calendar_provider=calendar_provider,
        gateway_factory=gateway_factory,
        snapshot_factory=snapshot_factory,
        recommendation_factory=recommendation_factory,
        now=korea_now,
        cache_stats=cache.stats_for_current_thread,
        progress=progress,
        warning=warning,
    )
    return BacktestRuntime(use_case, cache)


def create_backtest_score_runtime(
    settings: AgentSettings, args: Any
) -> BacktestScoreRuntime:
    """저장 결과, 실제 조회 게이트웨이, cache를 채점 유스케이스에 연결한다."""

    output_dir = Path(args.result_dir)
    result_store = FileBacktestResultStore(output_dir)
    manifest = result_store.read_manifest()
    if manifest is None:
        raise ValueError("manifest.json을 찾을 수 없습니다.")
    arguments = manifest.get("arguments", {})
    cache_from_manifest = (
        arguments.get("cache_dir") if isinstance(arguments, dict) else None
    )
    cache_dir = Path(args.cache_dir or cache_from_manifest or "data/backtest/cache")
    output_json = Path(args.output or output_dir / "score.json")
    output_markdown = output_json.with_suffix(".md")
    gateway = TradingApiMarketDataGateway(settings.trading_api)
    cache = ResponseCache(gateway.client, cache_dir)
    gateway.client = cache  # type: ignore[assignment]
    options = BacktestScoreOptions(
        horizons=args.horizons,
        policy=ConsensusPolicyName(args.policy) if args.policy else None,
        confidence_aggregation=(
            ConfidenceAggregation(args.confidence_aggregation)
            if args.confidence_aggregation
            else None
        ),
        compare_policies=args.compare_policies,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        hold_bands=args.hold_bands,
    )
    use_case = ScoreBacktest(result_store, gateway, options)
    return BacktestScoreRuntime(use_case, gateway, output_json, output_markdown)


def create_recommendation_runtime(
    settings: AgentSettings,
    db_path: str | Path | None = None,
) -> RecommendationRuntime:
    """계좌를 읽지 않는 종목 추천 유스케이스를 조립한다."""

    clock = korea_now
    gateway = TradingApiMarketDataGateway(settings.trading_api)
    runner = SubprocessCommandRunner()
    service = RecommendationConsensusService(
        ClaudeCliDecisionModel(settings.llm.claude, runner, RECOMMENDATION_TASK),
        CodexCliDecisionModel(settings.llm.codex, runner, RECOMMENDATION_TASK),
        policy=settings.consensus.policy,
        confidence_aggregation=settings.consensus.confidence_aggregation,
    )
    use_case = RecommendStocks(
        snapshot_builder=SnapshotBuilder(gateway, settings.watchlist, clock=clock),
        consensus_service=service,
        universe=settings.recommendation_universe,
        ledger=SQLiteDecisionLedger(
            db_path or settings.ledger.db_path,
            stale_run_after=timedelta(seconds=settings.ledger.stale_run_after_seconds),
        ),
        clock=clock,
    )
    return RecommendationRuntime(use_case=use_case, gateway=gateway)


def _parse_dates_count(value: str | None) -> str | int:
    if value is None or value == "all":
        return "all"
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError("--dates는 all 또는 양의 정수여야 합니다.") from error
    if count <= 0:
        raise ValueError("--dates는 all 또는 양의 정수여야 합니다.")
    return count


def _parse_positive_list(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("--horizons는 양의 정수 목록이어야 합니다.") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("--horizons는 양의 정수 목록이어야 합니다.")
    return parsed


def _explicit_dates(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    dates = [item.strip() for value in values for item in value.split(",")]
    try:
        if any(len(item) != 8 or not item.isdigit() for item in dates):
            raise ValueError("--date 형식 오류")
        for item in dates:
            datetime.strptime(item, "%Y%m%d")
    except ValueError as error:
        raise ValueError("--date는 유효한 YYYYMMDD 날짜여야 합니다.") from error
    return dates


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()
