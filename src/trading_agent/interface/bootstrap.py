"""실행 설정에서 외부 어댑터와 사이클 유스케이스를 조립한다."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_agent.application.consensus_service import (
    ConsensusDecisionService,
    RecommendationConsensusService,
)
from trading_agent.application.model_tasks import DECISION_TASK, RECOMMENDATION_TASK
from trading_agent.application.recommend_stocks import RecommendStocks
from trading_agent.application.records import (
    DecisionCycleResult,
    RecommendationRunResult,
)
from trading_agent.application.run_decision_cycle import RunDecisionCycle
from trading_agent.application.snapshot_builder import SnapshotBuilder
from trading_agent.domain.risk import RiskPolicy
from trading_agent.infrastructure.command_runner import SubprocessCommandRunner
from trading_agent.infrastructure.config import AgentSettings
from trading_agent.infrastructure.decision_models import (
    ClaudeCliDecisionModel,
    CodexCliDecisionModel,
)
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
    )
    risk_policy = RiskPolicy(
        max_position_weight=Decimal(str(settings.risk.max_position_weight)),
        max_order_amount=Decimal(str(settings.risk.max_order_amount)),
        min_confidence=settings.risk.min_confidence,
        allowed_codes=set(settings.watchlist),
        max_proposals_per_cycle=settings.risk.max_proposals_per_cycle,
    )
    cycle = RunDecisionCycle(
        snapshot_builder=SnapshotBuilder(gateway, settings.watchlist, clock=clock),
        consensus_service=consensus_service,
        risk_policy=risk_policy,
        watchlist=settings.watchlist,
        ledger=SQLiteDecisionLedger(db_path or settings.ledger.db_path),
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
    )
    use_case = RecommendStocks(
        snapshot_builder=SnapshotBuilder(gateway, settings.watchlist, clock=clock),
        consensus_service=service,
        universe=settings.recommendation_universe,
        ledger=SQLiteDecisionLedger(db_path or settings.ledger.db_path),
        clock=clock,
    )
    return RecommendationRuntime(use_case=use_case, gateway=gateway)
