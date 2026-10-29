"""사이클 실행 기록 저장 포트."""

from abc import ABC, abstractmethod
from datetime import datetime

from trading_agent.application.records import DecisionCycleResult


class DecisionLedger(ABC):
    """사이클 시작과 종료 결과를 저장하는 포트."""

    @abstractmethod
    def start_cycle(self, started_at: datetime) -> int:
        """새 사이클을 시작 상태로 저장하고 식별자를 반환한다."""

    @abstractmethod
    def complete_cycle(
        self, result: DecisionCycleResult, finished_at: datetime
    ) -> None:
        """완료한 사이클과 그 상세 결과를 저장한다."""

    @abstractmethod
    def fail_cycle(self, result: DecisionCycleResult, finished_at: datetime) -> None:
        """실패한 사이클과 확보된 상세 결과를 저장한다."""
