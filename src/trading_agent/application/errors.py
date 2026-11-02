"""애플리케이션 계층에서 사용하는 오류."""

from trading_agent.application.ports import ModelRun


class MarketDataUnavailable(RuntimeError):
    """시장 데이터 요청이나 응답 변환에 실패했다."""

    def __init__(self, message: str, reason: str = "unavailable") -> None:
        super().__init__(message)
        self.reason = reason


class DecisionModelError(RuntimeError):
    """모델 판단이 실패했지만 실행 원문은 보존되어 있다."""

    def __init__(self, message: str, run: ModelRun) -> None:
        super().__init__(message)
        self.run = run
