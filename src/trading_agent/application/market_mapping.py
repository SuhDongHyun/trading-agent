"""시장명에서 종목의 지수 벤치마크를 판별한다."""

from trading_agent.application.ports import MarketIndex


def market_index_for(market_name: str | None) -> MarketIndex | None:
    """응답의 시장명으로 종목 벤치마크를 고른다."""

    normalized = (market_name or "").strip().upper()
    if (
        normalized.startswith("KOSPI")
        or "코스피" in normalized
        or "유가증권" in normalized
    ):
        return MarketIndex.KOSPI
    if normalized.startswith(("KOSDAQ", "KSQ")) or "코스닥" in normalized:
        return MarketIndex.KOSDAQ
    return None
