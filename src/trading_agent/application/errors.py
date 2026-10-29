"""애플리케이션 계층에서 사용하는 오류."""


class MarketDataUnavailable(RuntimeError):
    """시장 데이터 요청이나 응답 변환에 실패했다."""
