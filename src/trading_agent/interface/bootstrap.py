"""CLI 인자와 설정에서 application 유스케이스·외부 어댑터를 조립한다."""

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from functools import partial
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
from trading_agent.application.paper_trading import (
    MarkPaperPortfolio,
    PaperCycleOutcome,
    PaperReport,
    PaperValuation,
    RunPaperCycle,
    build_paper_report,
)
from trading_agent.application.ports import MarketDataGateway, MarketIndex
from trading_agent.application.prompt_builder import build_decision_prompt
from trading_agent.application.recommend_stocks import RecommendStocks
from trading_agent.application.recommendation_prompt import (
    build_recommendation_prompt,
)
from trading_agent.application.records import (
    DecisionCycleResult,
    RecommendationRunResult,
)
from trading_agent.application.run_decision_cycle import RunDecisionCycle
from trading_agent.application.snapshot_builder import SnapshotBuilder
from trading_agent.domain.models import ConfidenceAggregation, ConsensusPolicyName
from trading_agent.domain.paper import FeeSchedule
from trading_agent.domain.price_guard import PriceGuardMode, PriceGuardPolicy
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
from trading_agent.infrastructure.sqlite_paper_store import SQLitePaperPortfolioStore
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

    gateway = TradingApiMarketDataGateway(settings.trading_api)
    cycle = _build_decision_cycle(
        settings, gateway, db_path or settings.ledger.db_path, force
    )
    return TradingAgentRuntime(cycle=cycle, gateway=gateway)


def _risk_policy(settings: AgentSettings) -> RiskPolicy:
    """설정의 리스크 한도를 도메인 정책으로 바꾼다."""

    return RiskPolicy(
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


def _build_decision_cycle(
    settings: AgentSettings,
    gateway: MarketDataGateway,
    db_path: str | Path,
    force: bool = False,
) -> RunDecisionCycle:
    """주어진 시장 데이터 포트로 판단 사이클을 조립한다."""

    clock = korea_now
    runner = SubprocessCommandRunner()
    consensus_service = ConsensusDecisionService(
        ClaudeCliDecisionModel(settings.llm.claude, runner, DECISION_TASK),
        CodexCliDecisionModel(settings.llm.codex, runner, DECISION_TASK),
        policy=settings.consensus.policy,
        confidence_aggregation=settings.consensus.confidence_aggregation,
        process_supervisor=runner,
    )
    return RunDecisionCycle(
        snapshot_builder=SnapshotBuilder(
            gateway,
            settings.watchlist,
            clock=clock,
            news_limit=settings.snapshot.news_limit,
            news_lookback_trading_days=settings.snapshot.news_lookback_trading_days,
        ),
        consensus_service=consensus_service,
        risk_policy=_risk_policy(settings),
        watchlist=settings.watchlist,
        ledger=SQLiteDecisionLedger(
            db_path,
            stale_run_after=timedelta(seconds=settings.ledger.stale_run_after_seconds),
        ),
        clock=clock,
        prompt_builder=partial(
            build_decision_prompt, horizon_days=settings.recommendation.horizon_days
        ),
        force=force,
    )


def _fee_schedule(settings: AgentSettings) -> FeeSchedule:
    paper = settings.paper
    return FeeSchedule(
        commission_rate=Decimal(str(paper.commission_rate)),
        sell_tax_rate=Decimal(str(paper.sell_tax_rate)),
        slippage_rate=Decimal(str(paper.slippage_rate)),
    )


@dataclass(slots=True)
class PaperRuntime[R]:
    """모의투자 유스케이스 실행과 HTTP 연결 정리를 제공한다."""

    use_case: Any
    gateway: TradingApiMarketDataGateway

    def run(self) -> R:
        return self.use_case.run()

    def close(self) -> None:
        self.gateway.close()


def create_paper_runtime(
    settings: AgentSettings, args: Any
) -> PaperRuntime[PaperCycleOutcome]:
    """가상 계좌로 판단 사이클을 돌리고 모의 체결하는 런타임을 만든다."""

    gateway = TradingApiMarketDataGateway(settings.trading_api)
    ledger_path = getattr(args, "ledger_db", None) or settings.paper.ledger_db_path
    if Path(ledger_path).resolve() == Path(settings.ledger.db_path).resolve():
        raise ValueError("모의투자 판단 기록은 운영 ledger와 다른 DB를 써야 합니다.")
    force = bool(getattr(args, "force", False))
    use_case = RunPaperCycle(
        store=SQLitePaperPortfolioStore(_paper_db(settings, args)),
        gateway=gateway,
        cycle_factory=lambda paper_gateway: _build_decision_cycle(
            settings, paper_gateway, ledger_path, force
        ),
        fees=_fee_schedule(settings),
        watchlist=settings.watchlist,
        initial_cash=Decimal(str(settings.paper.initial_cash)),
        clock=korea_now,
        reuse_enabled=settings.paper.reuse_decision.enabled,
        price_move_atr_multiple=Decimal(
            str(settings.paper.reuse_decision.price_move_atr_multiple)
        ),
        force_no_reuse=bool(getattr(args, "no_reuse", False)),
    )
    return PaperRuntime(use_case=use_case, gateway=gateway)


def create_paper_mark_runtime(
    settings: AgentSettings, args: Any
) -> PaperRuntime[PaperValuation]:
    """매매 없이 가상 계좌를 평가하는 런타임을 만든다."""

    gateway = TradingApiMarketDataGateway(settings.trading_api)
    use_case = MarkPaperPortfolio(
        store=SQLitePaperPortfolioStore(_paper_db(settings, args)),
        gateway=gateway,
        watchlist=settings.watchlist,
        initial_cash=Decimal(str(settings.paper.initial_cash)),
        clock=korea_now,
    )
    return PaperRuntime(use_case=use_case, gateway=gateway)


def load_paper_report(settings: AgentSettings, args: Any) -> PaperReport | None:
    """저장된 가상 계좌 이력으로 리포트를 계산한다. 계좌가 없으면 None."""

    store = SQLitePaperPortfolioStore(_paper_db(settings, args))
    portfolio = store.load()
    if portfolio is None:
        return None
    return build_paper_report(
        portfolio,
        store.history(),
        settings.watchlist,
        recent_fills=getattr(args, "fills", 10),
    )


def _paper_db(settings: AgentSettings, args: Any) -> Path:
    return Path(getattr(args, "paper_db", None) or settings.paper.db_path)


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
        available_horizons = [int(value) for value in scored.data["horizons"]]
        primary_horizons = self.use_case.options.primary_horizons
        primary = [value for value in primary_horizons if value in available_horizons]
        ordered_horizons = primary + [
            value for value in available_horizons if value not in primary
        ]
        for horizon in ordered_horizons:
            metric = scored.data["sources"]["consensus"]["horizons"][str(horizon)]
            label = (
                f"{horizon}일" if horizon in primary_horizons else f"{horizon}일(참고)"
            )
            print(
                f"합의 {label}: buy {metric['buy_n']}건, "
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
        dates_explicit=args.dates is not None,
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
        top_candidates=settings.recommendation.top_candidates,
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
        builder = SnapshotBuilder(
            point_gateway,
            codes,
            clock=clock,
            news_limit=settings.snapshot.news_limit,
            news_lookback_trading_days=settings.snapshot.news_lookback_trading_days,
        )
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
            process_supervisor=runner,
        )
        assert shared_ledger is not None
        return RecommendStocks(
            snapshot_builder=snapshot_builder,  # type: ignore[arg-type]
            consensus_service=service,
            universe=codes,
            ledger=shared_ledger,
            clock=clock,
            prompt_builder=partial(
                build_recommendation_prompt,
                horizon_days=settings.recommendation.horizon_days,
            ),
            price_guard_policy=_price_guard_policy(settings),
            top_candidates=settings.recommendation.top_candidates,
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
        process_supervisor=runner,
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
        top_candidates=args.top_candidates,
        default_top_candidates=settings.recommendation.top_candidates,
        primary_horizons=tuple(settings.recommendation.horizon_days),
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
    return BacktestScoreRuntime(
        use_case,
        gateway,
        output_json,
        output_markdown,
    )


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
        process_supervisor=runner,
    )
    use_case = RecommendStocks(
        snapshot_builder=SnapshotBuilder(
            gateway,
            settings.watchlist,
            clock=clock,
            news_limit=settings.snapshot.news_limit,
            news_lookback_trading_days=settings.snapshot.news_lookback_trading_days,
        ),
        consensus_service=service,
        universe=settings.recommendation_universe,
        ledger=SQLiteDecisionLedger(
            db_path or settings.ledger.db_path,
            stale_run_after=timedelta(seconds=settings.ledger.stale_run_after_seconds),
        ),
        clock=clock,
        prompt_builder=partial(
            build_recommendation_prompt,
            horizon_days=settings.recommendation.horizon_days,
        ),
        price_guard_policy=_price_guard_policy(settings),
        top_candidates=settings.recommendation.top_candidates,
    )
    return RecommendationRuntime(use_case=use_case, gateway=gateway)


def _price_guard_policy(settings: AgentSettings) -> PriceGuardPolicy:
    """설정 값으로 도메인 추천 가격 가드를 만든다."""

    return PriceGuardPolicy(
        mode=PriceGuardMode(settings.price_guard.mode),
        min_stop_atr_multiple=Decimal(str(settings.price_guard.min_stop_atr_multiple)),
        min_reward_risk=Decimal(str(settings.price_guard.min_reward_risk)),
    )


def _parse_dates_count(value: str | None) -> str | int:
    if value is None:
        return 10
    if value == "all":
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
