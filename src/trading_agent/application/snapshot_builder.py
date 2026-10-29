"""시장 데이터 조회 결과를 LLM 입력용 스냅샷으로 조립한다."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from trading_agent.application.errors import MarketDataUnavailable
from trading_agent.application.ports import (
    IndexPoint,
    IndicatorPoint,
    MarketDataGateway,
    MarketIndex,
    StockQuote,
    TreasuryMaturity,
)
from trading_agent.domain.models import (
    AccountSnapshot,
    MarketContext,
    MarketSnapshot,
    StockSnapshot,
)

LOOKBACK_CALENDAR_DAYS = 120
MA_SHORT_WINDOW = 20
MA_LONG_WINDOW = 60
RSI_WINDOW = 14
RSI_SIGNAL_WINDOW = 9
MACD_SHORT_WINDOW = 12
MACD_LONG_WINDOW = 26
MACD_SIGNAL_WINDOW = 9
VOLUME_BASELINE_WINDOW = 20
DEFAULT_NEWS_LIMIT = 5
MARKET_TIMEZONE = ZoneInfo("Asia/Seoul")


def calculate_return_percent(closes: Sequence[Decimal], periods: int) -> Decimal | None:
    """최근 종가와 지정 거래일 이전 종가 사이의 수익률을 계산한다."""

    if periods <= 0:
        raise ValueError("periods는 0보다 커야 합니다.")
    if len(closes) <= periods:
        return None
    previous_close = closes[-(periods + 1)]
    if previous_close == 0:
        return None
    return (closes[-1] - previous_close) / previous_close * Decimal(100)


def calculate_deviation_percent(
    current_price: Decimal, moving_average: Decimal | None
) -> Decimal | None:
    """현재가와 이동평균 사이의 백분율 괴리를 계산한다."""

    if moving_average is None or moving_average == 0:
        return None
    return (current_price - moving_average) / moving_average * Decimal(100)


def calculate_volume_change_percent(
    volumes: Sequence[int], window: int = VOLUME_BASELINE_WINDOW
) -> Decimal | None:
    """최신 거래량을 직전 구간의 평균 거래량과 비교한다."""

    if window <= 0:
        raise ValueError("window는 0보다 커야 합니다.")
    if len(volumes) <= window:
        return None
    baseline = sum(volumes[-(window + 1) : -1])
    if baseline == 0:
        return None
    average_volume = Decimal(baseline) / Decimal(window)
    return (Decimal(volumes[-1]) - average_volume) / average_volume * Decimal(100)


def _seoul_now() -> datetime:
    """trading-api와 같은 한국 날짜의 현재 시각을 반환한다."""

    return datetime.now(MARKET_TIMEZONE)


class SnapshotBuilder:
    """계좌·종목·시장 데이터를 모아 한 번의 시장 스냅샷을 만든다."""

    def __init__(
        self,
        gateway: MarketDataGateway,
        watchlist: Sequence[str],
        clock: Callable[[], datetime] = _seoul_now,
        news_limit: int = DEFAULT_NEWS_LIMIT,
    ) -> None:
        if news_limit <= 0:
            raise ValueError("news_limit는 0보다 커야 합니다.")
        self.gateway = gateway
        self.watchlist = tuple(watchlist)
        self.clock = clock
        self.news_limit = news_limit

    def build(self) -> MarketSnapshot:
        """필수 데이터가 있으면 스냅샷을 만들고 보조 실패는 기록한다."""

        now = self.clock()
        today = now.date()
        end_date = today.strftime("%Y%m%d")
        start_date = (today - timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime("%Y%m%d")

        account = self.gateway.get_account()
        codes = _unique_codes(
            [*self.watchlist, *(position.code for position in account.positions)]
        )
        quotes = {code: self.gateway.get_stock_quote(code) for code in codes}
        valued_account = _with_current_prices(account, quotes)
        missing: list[str] = []

        stocks = [
            self._build_stock(
                code,
                quotes[code],
                start_date,
                end_date,
                missing,
            )
            for code in codes
        ]
        market_context = self._build_market_context(start_date, end_date, missing)
        return MarketSnapshot(
            created_at=now,
            account=valued_account,
            stocks=stocks,
            market_context=market_context,
            missing=missing,
        )

    def _build_stock(
        self,
        code: str,
        quote: StockQuote,
        start_date: str,
        end_date: str,
        missing: list[str],
    ) -> StockSnapshot:
        """한 종목의 필수 현재가와 보조 시계열을 요약한다."""

        daily = self._read_optional(
            f"{code}:daily",
            lambda: self.gateway.get_daily_bars(code, start_date, end_date),
            missing,
        )
        ma20 = self._latest_optional(
            f"{code}:ma20",
            lambda: self.gateway.get_moving_average(
                code, start_date, end_date, MA_SHORT_WINDOW
            ),
            missing,
        )
        ma60 = self._latest_optional(
            f"{code}:ma60",
            lambda: self.gateway.get_moving_average(
                code, start_date, end_date, MA_LONG_WINDOW
            ),
            missing,
        )
        rsi = self._latest_optional(
            f"{code}:rsi",
            lambda: self.gateway.get_rsi(code, start_date, end_date, RSI_WINDOW),
            missing,
        )
        rsi_signal = self._latest_optional(
            f"{code}:rsi_signal",
            lambda: self.gateway.get_rsi_signal(
                code,
                start_date,
                end_date,
                RSI_WINDOW,
                RSI_SIGNAL_WINDOW,
            ),
            missing,
        )
        macd = self._latest_optional(
            f"{code}:macd",
            lambda: self.gateway.get_macd(
                code,
                start_date,
                end_date,
                MACD_SHORT_WINDOW,
                MACD_LONG_WINDOW,
            ),
            missing,
        )
        macd_signal = self._latest_optional(
            f"{code}:macd_signal",
            lambda: self.gateway.get_macd_signal(
                code,
                start_date,
                end_date,
                MACD_SHORT_WINDOW,
                MACD_LONG_WINDOW,
                MACD_SIGNAL_WINDOW,
            ),
            missing,
        )
        news = self._read_optional(
            f"{code}:news",
            lambda: self.gateway.get_stock_news(code, end_date),
            missing,
        )

        sorted_daily = sorted(daily or [], key=lambda item: item.date)
        if not sorted_daily:
            _mark_missing(f"{code}:daily", missing)
        sorted_closes = [bar.close_price for bar in sorted_daily]
        sorted_volumes = [bar.volume for bar in sorted_daily]
        latest_ma20 = ma20.value if ma20 else None
        latest_ma60 = ma60.value if ma60 else None
        headlines = None
        if news is not None:
            headlines = [
                item.title
                for item in sorted(
                    news, key=lambda item: item.published_at, reverse=True
                )[: self.news_limit]
            ]

        return StockSnapshot(
            code=code,
            name=quote.name,
            current_price=quote.current_price,
            change_percent=quote.change_percent,
            return_5d_percent=calculate_return_percent(sorted_closes, 5),
            return_20d_percent=calculate_return_percent(sorted_closes, 20),
            ma20_deviation_percent=calculate_deviation_percent(
                quote.current_price, latest_ma20
            ),
            ma60_deviation_percent=calculate_deviation_percent(
                quote.current_price, latest_ma60
            ),
            rsi=rsi.value if rsi else None,
            rsi_signal=rsi_signal.signal if rsi_signal else None,
            macd=macd.value if macd else None,
            macd_signal_value=macd_signal.value if macd_signal else None,
            macd_signal=macd_signal.signal if macd_signal else None,
            volume_change_percent=calculate_volume_change_percent(sorted_volumes),
            news_headlines=headlines,
        )

    def _build_market_context(
        self, start_date: str, end_date: str, missing: list[str]
    ) -> MarketContext:
        """시장 시계열별 최신값과 최근 등락을 구성한다."""

        kospi = self._latest_optional(
            "market:kospi",
            lambda: self.gateway.get_market_index(
                MarketIndex.KOSPI, start_date, end_date
            ),
            missing,
        )
        kosdaq = self._latest_optional(
            "market:kosdaq",
            lambda: self.gateway.get_market_index(
                MarketIndex.KOSDAQ, start_date, end_date
            ),
            missing,
        )
        sp500_points = self._read_optional(
            "market:sp500",
            lambda: self.gateway.get_market_index(
                MarketIndex.SP500, start_date, end_date
            ),
            missing,
        )
        sp500_change = _latest_change_percent(sp500_points or [])
        if sp500_points is not None and sp500_change is None:
            _mark_missing("market:sp500", missing)

        vix = self._latest_optional(
            "market:vix",
            lambda: self.gateway.get_market_index(
                MarketIndex.VIX, start_date, end_date
            ),
            missing,
        )
        vkospi = self._latest_optional(
            "market:vkospi",
            lambda: self.gateway.get_market_index(
                MarketIndex.VKOSPI, start_date, end_date
            ),
            missing,
        )
        fear_greed = self._read_optional(
            "market:fear_and_greed",
            self.gateway.get_fear_and_greed_index,
            missing,
        )
        exchange_rate = self._latest_optional(
            "market:usd_krw",
            lambda: self.gateway.get_market_index(
                MarketIndex.USD_KRW, start_date, end_date
            ),
            missing,
        )
        korea_3y = self._latest_optional(
            "market:korea_3y_yield",
            lambda: self.gateway.get_treasury_yield(
                TreasuryMaturity.KOREA_3Y, start_date, end_date
            ),
            missing,
        )
        us_10y = self._latest_optional(
            "market:us_10y_yield",
            lambda: self.gateway.get_treasury_yield(
                TreasuryMaturity.US_10Y, start_date, end_date
            ),
            missing,
        )
        return MarketContext(
            kospi_change_percent=kospi.change_percent if kospi else None,
            kosdaq_change_percent=kosdaq.change_percent if kosdaq else None,
            sp500_change_percent=sp500_change,
            volatility_index=vix.value if vix else None,
            vkospi=vkospi.value if vkospi else None,
            fear_greed_index=fear_greed.value if fear_greed else None,
            exchange_rate=exchange_rate.value if exchange_rate else None,
            korea_3y_yield=korea_3y.value if korea_3y else None,
            us_10y_yield=us_10y.value if us_10y else None,
        )

    @staticmethod
    def _read_optional[T](
        label: str,
        read: Callable[[], T],
        missing: list[str],
    ) -> T | None:
        """보조 조회 실패를 missing 항목으로 기록하고 계속한다."""

        try:
            return read()
        except MarketDataUnavailable:
            _mark_missing(label, missing)
            return None

    @classmethod
    def _latest_optional(
        cls,
        label: str,
        read: Callable[[], Sequence[IndicatorPoint] | Sequence[IndexPoint]],
        missing: list[str],
    ) -> IndicatorPoint | IndexPoint | None:
        """시계열의 최신 값을 가져오고 오류나 빈 응답을 기록한다."""

        points = cls._read_optional(label, read, missing)
        if not points:
            _mark_missing(label, missing)
            return None
        return max(points, key=lambda point: point.date)


def _unique_codes(codes: Sequence[str]) -> list[str]:
    """입력 순서를 보존하며 중복 종목 코드를 제거한다."""

    return list(dict.fromkeys(codes))


def _with_current_prices(
    account: AccountSnapshot, quotes: dict[str, StockQuote]
) -> AccountSnapshot:
    """계좌 보유 종목에 필수 현재가를 채운다."""

    positions = [
        replace(
            position,
            current_price=quotes[position.code].current_price,
        )
        for position in account.positions
    ]
    return AccountSnapshot(cash=account.cash, positions=positions)


def _latest_change_percent(points: Sequence[IndexPoint]) -> Decimal | None:
    """지수 응답의 등락률 또는 최근 두 종가의 변화율을 반환한다."""

    if not points:
        return None
    latest = max(points, key=lambda point: point.date)
    if latest.change_percent is not None:
        return latest.change_percent
    earlier = [point for point in points if point.date < latest.date]
    if not earlier:
        return None
    previous = max(earlier, key=lambda point: point.date)
    if previous.value == 0:
        return None
    return (latest.value - previous.value) / previous.value * Decimal(100)


def _mark_missing(label: str, missing: list[str]) -> None:
    """missing 목록에 같은 항목이 중복 추가되지 않도록 한다."""

    if label not in missing:
        missing.append(label)
