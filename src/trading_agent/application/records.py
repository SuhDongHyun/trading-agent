"""사이클 실행 결과와 저장 상태."""

from dataclasses import dataclass
from enum import StrEnum

from trading_agent.application.consensus_service import ConsensusDecisionResult
from trading_agent.domain.models import MarketSnapshot
from trading_agent.domain.risk import RiskVerdict


class CycleStatus(StrEnum):
    """사이클을 종료한 상태."""

    COMPLETED = "completed"
    MODEL_FAILED = "model_failed"
    DATA_FAILED = "data_failed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DecisionCycleResult:
    """한 사이클의 결과와 기록에 필요한 입력을 보존한다."""

    run_id: int
    status: CycleStatus
    error: str | None = None
    prompt: str | None = None
    snapshot: MarketSnapshot | None = None
    decision: ConsensusDecisionResult | None = None
    verdicts: list[RiskVerdict] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdicts", list(self.verdicts or []))
