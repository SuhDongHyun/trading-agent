"""과거 판단 시점에 사용할 시장 데이터 게이트웨이."""

from collections import Counter
from collections.abc import Callable
from datetime import datetime, time
from decimal import Decimal
from threading import Lock
from zoneinfo import ZoneInfo

from trading_agent.application.errors import MarketDataUnavailable
from trading_agent.application.ports import (
    DailyBar,
    IndexPoint,
    IndicatorPoint,
    MarketDataGateway,
    MarketIndex,
    NewsItem,
    StockQuote,
    TreasuryMaturity,
)
from trading_agent.domain.models import AccountSnapshot

KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


class SharedDailyCache:
    """여러 판단일 게이트웨이가 종목별 전체 일봉 응답을 공유한다."""

    def __init__(self) -> None:
        self.values: dict[str, list[DailyBar]] = {}
        self.lock = Lock()


class PointInTimeGateway(MarketDataGateway):
    """기준일 이후 데이터를 차단하는 ``MarketDataGateway`` 래퍼."""

    def __init__(
        self,
        inner: MarketDataGateway,
        decision_date: str,
        range_start: str,
        range_end: str,
        decision_time: time = time(15, 40),
        daily_cache: SharedDailyCache | None = None,
        account: AccountSnapshot | None = None,
    ) -> None:
        self.inner = inner
        self.decision_date = decision_date
        self.range_start = range_start
        self.range_end = range_end
        self.decision_at = datetime.fromisoformat(
            f"{decision_date[:4]}-{decision_date[4:6]}-{decision_date[6:]}T"
            f"{decision_time.isoformat()}"
        ).replace(tzinfo=KOREA_TIMEZONE)
        self.daily_cache = daily_cache or SharedDailyCache()
        self.account = account or AccountSnapshot(Decimal(0), [])
        self.leakage: Counter[str] = Counter()
        self._counted_daily_codes: set[str] = set()

    def get_account(self) -> AccountSnapshot:
        """백테스트 추천은 계좌를 쓰지 않고 주입 계좌만 반환한다."""

        return self.account

    def get_stock_quote(self, code: str) -> StockQuote:
        """기준일 종가와 직전 종가로 가격 및 등락률을 재구성한다."""

        actual = self.inner.get_stock_quote(code)
        bars = self.get_daily_bars(code, self.range_start, self.decision_date)
        dated = sorted(
            (bar for bar in bars if bar.date <= self.decision_date),
            key=lambda b: b.date,
        )
        today = next(
            (bar for bar in reversed(dated) if bar.date == self.decision_date), None
        )
        if today is None:
            raise MarketDataUnavailable(
                f"{code}: 기준일 일봉을 찾을 수 없습니다.",
                reason="no decision-date bar",
            )
        previous = next(
            (bar for bar in reversed(dated) if bar.date < self.decision_date), None
        )
        if previous is None or previous.close_price == 0:
            raise MarketDataUnavailable(
                f"{code}: 직전 거래일 종가를 찾을 수 없습니다.",
                reason="no previous close",
            )
        change = (
            (today.close_price - previous.close_price)
            / previous.close_price
            * Decimal(100)
        )
        return StockQuote(code, actual.name, today.close_price, change)

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        """종목별 전체 요청 구간을 한 번 받아 기준일까지만 돌려준다."""

        with self.daily_cache.lock:
            if code not in self.daily_cache.values:
                self.daily_cache.values[code] = self.inner.get_daily_bars(
                    code, self.range_start, self.range_end
                )
            bars = self.daily_cache.values[code]
        if code not in self._counted_daily_codes:
            self.leakage["daily_rows"] += sum(
                bar.date > self.decision_date for bar in bars
            )
            self._counted_daily_codes.add(code)
        cutoff = min(end_date, self.decision_date)
        return [bar for bar in bars if start_date <= bar.date <= cutoff]

    def get_moving_average(
        self, code: str, start_date: str, end_date: str, window: int
    ) -> list[IndicatorPoint]:
        return self._indicator(
            "moving_average",
            lambda: self.inner.get_moving_average(
                code, start_date, min(end_date, self.decision_date), window
            ),
        )

    def get_rsi(
        self, code: str, start_date: str, end_date: str, window: int = 14
    ) -> list[IndicatorPoint]:
        return self._indicator(
            "rsi",
            lambda: self.inner.get_rsi(
                code, start_date, min(end_date, self.decision_date), window
            ),
        )

    def get_rsi_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        rsi_window: int = 14,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        return self._indicator(
            "rsi_signal",
            lambda: self.inner.get_rsi_signal(
                code,
                start_date,
                min(end_date, self.decision_date),
                rsi_window,
                ema_window,
            ),
        )

    def get_macd(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
    ) -> list[IndicatorPoint]:
        return self._indicator(
            "macd",
            lambda: self.inner.get_macd(
                code,
                start_date,
                min(end_date, self.decision_date),
                short_window,
                long_window,
            ),
        )

    def get_macd_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        return self._indicator(
            "macd_signal",
            lambda: self.inner.get_macd_signal(
                code,
                start_date,
                min(end_date, self.decision_date),
                short_window,
                long_window,
                ema_window,
            ),
        )

    def _indicator(
        self, name: str, read: Callable[[], list[IndicatorPoint]]
    ) -> list[IndicatorPoint]:
        points = read()
        self.leakage[f"{name}_rows"] += sum(
            point.date > self.decision_date for point in points
        )
        return [point for point in points if point.date <= self.decision_date]

    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
        """미래 검색일은 건너뛰고 판단 시각 이후의 기사를 제거한다."""

        if search_date > self.decision_date:
            return []
        items = self.inner.get_stock_news(code, search_date)
        allowed: list[NewsItem] = []
        for item in items:
            published_at = item.published_at
            if published_at.tzinfo is None or published_at.utcoffset() is None:
                published_at = published_at.replace(tzinfo=KOREA_TIMEZONE)
            else:
                published_at = published_at.astimezone(KOREA_TIMEZONE)
            if published_at <= self.decision_at:
                allowed.append(item)
            else:
                self.leakage["news_articles"] += 1
        return allowed

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        """한국 지표는 기준일 포함, 미국 지표는 기준일 이전만 허용한다."""

        inclusive = index in (MarketIndex.KOSPI, MarketIndex.KOSDAQ, MarketIndex.VKOSPI)
        cutoff = (
            self.decision_date
            if inclusive
            else _previous_calendar_day(self.decision_date)
        )
        points = self.inner.get_market_index(index, start_date, min(end_date, cutoff))
        return self._market_points(f"market_{index.value}", points, start_date, cutoff)

    def get_fear_and_greed_index(self) -> IndexPoint:
        """최신 현재값은 과거 시점 데이터로 사용할 수 없다."""

        raise MarketDataUnavailable(
            "공포탐욕 지수는 과거 시점 데이터가 없습니다.", reason="not point-in-time"
        )

    def get_treasury_yield(
        self, maturity: TreasuryMaturity, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        """한국 금리는 기준일 포함, 미국 금리는 기준일 이전만 허용한다."""

        cutoff = (
            self.decision_date
            if maturity is TreasuryMaturity.KOREA_3Y
            else _previous_calendar_day(self.decision_date)
        )
        points = self.inner.get_treasury_yield(
            maturity, start_date, min(end_date, cutoff)
        )
        return self._market_points(
            f"treasury_{maturity.value}", points, start_date, cutoff
        )

    def _market_points(
        self, name: str, points: list[IndexPoint], start_date: str, cutoff: str
    ) -> list[IndexPoint]:
        self.leakage[f"{name}_rows"] += sum(point.date > cutoff for point in points)
        return [point for point in points if start_date <= point.date <= cutoff]


def _previous_calendar_day(value: str) -> str:
    """YYYYMMDD 문자열의 직전 달력 날짜를 반환한다."""

    date = datetime.strptime(value, "%Y%m%d").date()
    return (date.fromordinal(date.toordinal() - 1)).strftime("%Y%m%d")
