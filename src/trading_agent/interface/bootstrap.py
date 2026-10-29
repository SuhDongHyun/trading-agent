"""실행 설정에서 외부 어댑터와 사이클 유스케이스를 조립한다."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_agent.application.consensus_service import ConsensusDecisionService
from trading_agent.application.records import DecisionCycleResult
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
) -> TradingAgentRuntime:
    """설정을 적용한 네트워크·LLM 어댑터와 사이클을 조립한다."""

    clock = korea_now
    gateway = TradingApiMarketDataGateway(settings.trading_api)
    runner = SubprocessCommandRunner()
    consensus_service = ConsensusDecisionService(
        ClaudeCliDecisionModel(settings.llm.claude, runner),
        CodexCliDecisionModel(settings.llm.codex, runner),
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
    )
    return TradingAgentRuntime(cycle=cycle, gateway=gateway)
