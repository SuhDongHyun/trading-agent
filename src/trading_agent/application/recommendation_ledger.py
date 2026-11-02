"""종목 추천 실행 기록 저장 포트."""

from abc import ABC, abstractmethod
from datetime import datetime

from trading_agent.application.records import RecommendationRunResult


class RecommendationLedger(ABC):
    """추천 실행의 시작과 종료 결과를 저장하는 포트."""

    @abstractmethod
    def start_recommendation(self, started_at: datetime) -> int:
        """새 추천 실행을 시작 상태로 저장하고 식별자를 반환한다."""

    @abstractmethod
    def finish_recommendation(
        self, result: RecommendationRunResult, finished_at: datetime
    ) -> None:
        """추천 실행 결과(실패 포함)와 확보된 상세 데이터를 저장한다."""
