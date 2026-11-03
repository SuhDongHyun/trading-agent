"""시장 데이터 조회 결과를 LLM 입력용 스냅샷으로 조립한다."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import partial
from zoneinfo import ZoneInfo

from trading_agent.application.errors import MarketDataUnavailable
from trading_agent.application.market_mapping import market_index_for
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
from trading_agent.domain.models import (
    AccountSnapshot,
    DataBasis,
    MarketContext,
    MarketSnapshot,
    NewsSnapshotItem,
    Position,
    ResearchSnapshot,
    StockSnapshot,
)

LOOKBACK_CALENDAR_DAYS = 380
INDICATOR_LOOKBACK_CALENDAR_DAYS = 120
MARKET_LOOKBACK_CALENDAR_DAYS = 14
MARKET_INDEX_LOOKBACK_CALENDAR_DAYS = 100
MA_SHORT_WINDOW = 20
MA_LONG_WINDOW = 60
RSI_WINDOW = 14
RSI_SIGNAL_WINDOW = 9
MACD_SHORT_WINDOW = 12
MACD_LONG_WINDOW = 26
MACD_SIGNAL_WINDOW = 9
VOLUME_BASELINE_WINDOW = 20
ATR_WINDOW = 14
REALIZED_VOLATILITY_WINDOW = 20
PRICE_RANGE_WINDOW = 20
WEEK_52_TRADING_DAYS = 252
DEFAULT_NEWS_LIMIT = 5
DEFAULT_NEWS_LOOKBACK_TRADING_DAYS = 3
MARKET_TIMEZONE = ZoneInfo("Asia/Seoul")
MARKET_CLOSE_TIME = time(15, 30)


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


def calculate_index_return_percent(
    points: Sequence[IndexPoint], periods: int
) -> Decimal | None:
    """정렬한 지수 종가에서 지정 거래일 수익률을 계산한다."""

    ordered = sorted(points, key=lambda point: point.date)
    return calculate_return_percent([point.value for point in ordered], periods)


def calculate_index_deviation_percent(
    points: Sequence[IndexPoint], window: int
) -> Decimal | None:
    """현재 지수와 최근 구간 단순 이동평균의 괴리율을 계산한다."""

    if window <= 0:
        raise ValueError("window는 0보다 커야 합니다.")
    ordered = sorted(points, key=lambda point: point.date)
    if len(ordered) < window:
        return None
    recent_values = [point.value for point in ordered[-window:]]
    moving_average = sum(recent_values, Decimal(0)) / Decimal(window)
    return calculate_deviation_percent(ordered[-1].value, moving_average)


def calculate_index_change_points(
    points: Sequence[IndexPoint], periods: int
) -> Decimal | None:
    """최근 지수 값과 지정 거래일 전 값의 포인트 차이를 계산한다."""

    if periods <= 0:
        raise ValueError("periods는 0보다 커야 합니다.")
    ordered = sorted(points, key=lambda point: point.date)
    if len(ordered) <= periods:
        return None
    return ordered[-1].value - ordered[-(periods + 1)].value


def calculate_universe_rank_percentile(
    value: Decimal | None, universe_values: Sequence[Decimal | None]
) -> Decimal | None:
    """동순위 평균을 사용해 universe 내 수익률 백분위를 계산한다."""

    if value is None:
        return None
    valid_values = [candidate for candidate in universe_values if candidate is not None]
    if len(valid_values) < 2:
        return None
    lower_count = sum(candidate < value for candidate in valid_values)
    equal_count = sum(candidate == value for candidate in valid_values)
    if equal_count == 0:
        return None
    average_rank_from_zero = Decimal(lower_count) + Decimal(equal_count - 1) / Decimal(
        2
    )
    return average_rank_from_zero / Decimal(len(valid_values) - 1) * Decimal(100)


def calculate_industry_relative_strength_20d_percent(
    stock_return_percent: Decimal | None,
    industry: str | None,
    peers: Sequence[tuple[str | None, Decimal | None]],
) -> Decimal | None:
    """같은 업종의 다른 종목 평균 대비 20일 초과수익을 계산한다."""

    normalized_industry = _normalize_industry(industry)
    if normalized_industry is None or stock_return_percent is None:
        return None
    peer_returns = [
        return_percent
        for peer_industry, return_percent in peers
        if _normalize_industry(peer_industry) == normalized_industry
        and return_percent is not None
    ]
    if not peer_returns:
        return None
    peer_average = sum(peer_returns, Decimal(0)) / Decimal(len(peer_returns))
    return stock_return_percent - peer_average


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


def calculate_atr14_percent(
    bars: Sequence[DailyBar], current_price: Decimal
) -> Decimal | None:
    """Wilder 방식 ATR(14)을 현재가 대비 백분율로 계산한다."""

    if current_price <= 0 or len(bars) <= ATR_WINDOW:
        return None
    if any(bar.high_price is None or bar.low_price is None for bar in bars):
        return None
    true_ranges: list[Decimal] = []
    for previous, bar in zip(bars, bars[1:], strict=False):
        assert bar.high_price is not None and bar.low_price is not None
        true_ranges.append(
            max(
                bar.high_price - bar.low_price,
                abs(bar.high_price - previous.close_price),
                abs(bar.low_price - previous.close_price),
            )
        )
    if len(true_ranges) < ATR_WINDOW:
        return None
    atr = sum(true_ranges[:ATR_WINDOW], Decimal(0)) / Decimal(ATR_WINDOW)
    for true_range in true_ranges[ATR_WINDOW:]:
        atr = (atr * Decimal(ATR_WINDOW - 1) + true_range) / Decimal(ATR_WINDOW)
    return atr / current_price * Decimal(100)


def calculate_realized_volatility_20d_percent(
    closes: Sequence[Decimal],
) -> Decimal | None:
    """최근 20개 단순 일간 수익률의 모집단 표준편차를 계산한다."""

    if len(closes) <= REALIZED_VOLATILITY_WINDOW:
        return None
    recent = closes[-(REALIZED_VOLATILITY_WINDOW + 1) :]
    previous_closes = recent[:-1]
    if any(close == 0 for close in previous_closes):
        return None
    returns = [
        (current - previous) / previous * Decimal(100)
        for previous, current in zip(recent, recent[1:], strict=False)
    ]
    mean = sum(returns, Decimal(0)) / Decimal(REALIZED_VOLATILITY_WINDOW)
    variance = sum(((value - mean) ** 2 for value in returns), Decimal(0)) / Decimal(
        REALIZED_VOLATILITY_WINDOW
    )
    return variance.sqrt()


def calculate_20d_price_distances(
    bars: Sequence[DailyBar], current_price: Decimal
) -> tuple[Decimal | None, Decimal | None]:
    """현재가에서 최근 20일 고가·저가까지의 부호 있는 거리를 계산한다."""

    if current_price <= 0 or len(bars) < PRICE_RANGE_WINDOW:
        return None, None
    recent = bars[-PRICE_RANGE_WINDOW:]
    if any(bar.high_price is None or bar.low_price is None for bar in recent):
        return None, None
    highs = [bar.high_price for bar in recent]
    lows = [bar.low_price for bar in recent]
    assert all(value is not None for value in highs + lows)
    highest = max(value for value in highs if value is not None)
    lowest = min(value for value in lows if value is not None)
    if highest <= 0 or lowest <= 0:
        return None, None
    high_distance = (current_price - highest) / highest * Decimal(100)
    low_distance = (current_price - lowest) / lowest * Decimal(100)
    return high_distance, low_distance


def calculate_52w_position(
    bars: Sequence[DailyBar], current_price: Decimal
) -> tuple[Decimal | None, int | None]:
    """가용한 최대 252거래일 고저 범위에서 현재가 위치를 계산한다."""

    if current_price <= 0 or not bars:
        return None, None
    recent = bars[-WEEK_52_TRADING_DAYS:]
    if any(bar.high_price is None or bar.low_price is None for bar in recent):
        return None, None
    highs = [bar.high_price for bar in recent]
    lows = [bar.low_price for bar in recent]
    assert all(value is not None for value in highs + lows)
    highest = max(value for value in highs if value is not None)
    lowest = min(value for value in lows if value is not None)
    if highest <= 0 or lowest <= 0 or highest == lowest:
        return None, None
    position = (current_price - lowest) / (highest - lowest) * Decimal(100)
    return position, len(recent)


def _atr_missing_reason(bars: Sequence[DailyBar], current_price: Decimal) -> str:
    if current_price <= 0:
        return "current price is zero"
    if len(bars) <= ATR_WINDOW:
        return "insufficient daily bars"
    if any(bar.high_price is None or bar.low_price is None for bar in bars):
        return "OHLC unavailable"
    return "invalid OHLC"


def _volatility_missing_reason(closes: Sequence[Decimal]) -> str:
    if len(closes) <= REALIZED_VOLATILITY_WINDOW:
        return "insufficient daily returns"
    return "zero previous close"


def _range_missing_reason(
    bars: Sequence[DailyBar],
    current_price: Decimal,
    window: int,
    *,
    allow_partial: bool = False,
) -> str:
    if current_price <= 0:
        return "current price is zero"
    if not bars:
        return "no daily bars"
    if not allow_partial and len(bars) < window:
        return "insufficient daily bars"
    recent = bars[-window:]
    if any(bar.high_price is None or bar.low_price is None for bar in recent):
        return "OHLC unavailable"
    return "invalid OHLC"


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
        news_lookback_trading_days: int = DEFAULT_NEWS_LOOKBACK_TRADING_DAYS,
    ) -> None:
        if (
            isinstance(news_limit, bool)
            or not isinstance(news_limit, int)
            or not 1 <= news_limit <= 20
        ):
            raise ValueError("news_limit는 1~20이어야 합니다.")
        if (
            isinstance(news_lookback_trading_days, bool)
            or not isinstance(news_lookback_trading_days, int)
            or not 1 <= news_lookback_trading_days <= 10
        ):
            raise ValueError("news_lookback_trading_days는 1~10이어야 합니다.")
        self.gateway = gateway
        self.watchlist = tuple(watchlist)
        self.clock = clock
        self.news_limit = news_limit
        self.news_lookback_trading_days = news_lookback_trading_days

    def build(self) -> MarketSnapshot:
        """필수 데이터가 있으면 스냅샷을 만들고 보조 실패는 기록한다."""

        (
            now,
            daily_start_date,
            indicator_start_date,
            end_date,
            market_start_date,
            market_index_start_date,
        ) = self._lookback_window()
        account = self.gateway.get_account()
        codes = _unique_codes(
            [*self.watchlist, *(position.code for position in account.positions)]
        )
        quotes = {code: self._required_quote(code) for code in codes}
        valued_account = _with_current_prices(account, quotes)
        missing: list[str] = []

        stock_results = [
            self._build_stock(
                code,
                quotes[code],
                now,
                daily_start_date,
                indicator_start_date,
                end_date,
                missing,
            )
            for code in codes
        ]
        stocks = [stock for stock, _ in stock_results]
        as_of_trading_date = _latest_daily_date(stock_results)
        market_context = self._build_market_context(
            market_index_start_date, market_start_date, end_date, missing
        )
        stocks = _add_relative_strength_fields(
            stocks,
            {code: quote.market_name for code, quote in quotes.items()},
            market_context,
        )
        return MarketSnapshot(
            created_at=now,
            account=valued_account,
            stocks=stocks,
            market_context=market_context,
            missing=missing,
            as_of_trading_date=as_of_trading_date,
            market_session=_market_session(now, as_of_trading_date, missing),
            data_basis=_data_basis(self.news_lookback_trading_days),
        )

    def build_research(self, codes: Sequence[str]) -> ResearchSnapshot:
        """계좌 없이 후보 종목과 시장 데이터만으로 스냅샷을 만든다.

        현재가를 못 읽은 종목은 missing에 적고 제외한다. 한 종목도 읽지 못하면 실패한다.
        """

        (
            now,
            daily_start_date,
            indicator_start_date,
            end_date,
            market_start_date,
            market_index_start_date,
        ) = self._lookback_window()
        missing: list[str] = []
        quotes: dict[str, StockQuote] = {}
        for code in _unique_codes(codes):
            quote = self._read_optional(
                f"{code}:quote", partial(self.gateway.get_stock_quote, code), missing
            )
            if quote is not None:
                quotes[code] = quote
        if not quotes:
            raise MarketDataUnavailable("후보 종목 현재가를 하나도 읽지 못했습니다.")

        stock_results = [
            self._build_stock(
                code,
                quote,
                now,
                daily_start_date,
                indicator_start_date,
                end_date,
                missing,
            )
            for code, quote in quotes.items()
        ]
        stocks = [stock for stock, _ in stock_results]
        as_of_trading_date = _latest_daily_date(stock_results)
        market_context = self._build_market_context(
            market_index_start_date, market_start_date, end_date, missing
        )
        stocks = _add_relative_strength_fields(
            stocks,
            {code: quote.market_name for code, quote in quotes.items()},
            market_context,
        )
        return ResearchSnapshot(
            created_at=now,
            stocks=stocks,
            market_context=market_context,
            missing=missing,
            as_of_trading_date=as_of_trading_date,
            market_session=_market_session(now, as_of_trading_date, missing),
            data_basis=_data_basis(self.news_lookback_trading_days),
        )

    def _lookback_window(self) -> tuple[datetime, str, str, str, str, str]:
        """현재 시각과 한국 날짜 기준 조회 시작일·종료일을 반환한다."""

        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock는 timezone-aware datetime을 반환해야 합니다.")
        today = now.astimezone(MARKET_TIMEZONE).date()
        end_date = today.strftime("%Y%m%d")
        daily_start_date = (today - timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime(
            "%Y%m%d"
        )
        indicator_start_date = (
            today - timedelta(days=INDICATOR_LOOKBACK_CALENDAR_DAYS)
        ).strftime("%Y%m%d")
        market_start_date = (
            today - timedelta(days=MARKET_LOOKBACK_CALENDAR_DAYS)
        ).strftime("%Y%m%d")
        market_index_start_date = (
            today - timedelta(days=MARKET_INDEX_LOOKBACK_CALENDAR_DAYS)
        ).strftime("%Y%m%d")
        return (
            now,
            daily_start_date,
            indicator_start_date,
            end_date,
            market_start_date,
            market_index_start_date,
        )

    def _build_stock(
        self,
        code: str,
        quote: StockQuote,
        created_at: datetime,
        daily_start_date: str,
        indicator_start_date: str,
        end_date: str,
        missing: list[str],
    ) -> tuple[StockSnapshot, str | None]:
        """한 종목의 필수 현재가와 보조 시계열을 요약한다."""

        daily = self._read_optional(
            f"{code}:daily",
            lambda: self.gateway.get_daily_bars(code, daily_start_date, end_date),
            missing,
        )
        ma20 = self._latest_optional(
            f"{code}:ma20",
            lambda: self.gateway.get_moving_average(
                code, indicator_start_date, end_date, MA_SHORT_WINDOW
            ),
            missing,
        )
        ma60 = self._latest_optional(
            f"{code}:ma60",
            lambda: self.gateway.get_moving_average(
                code, indicator_start_date, end_date, MA_LONG_WINDOW
            ),
            missing,
        )
        rsi = self._latest_optional(
            f"{code}:rsi",
            lambda: self.gateway.get_rsi(
                code, indicator_start_date, end_date, RSI_WINDOW
            ),
            missing,
        )
        rsi_signal = self._latest_optional(
            f"{code}:rsi_signal",
            lambda: self.gateway.get_rsi_signal(
                code,
                indicator_start_date,
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
                indicator_start_date,
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
                indicator_start_date,
                end_date,
                MACD_SHORT_WINDOW,
                MACD_LONG_WINDOW,
                MACD_SIGNAL_WINDOW,
            ),
            missing,
        )
        sorted_daily = sorted(daily or [], key=lambda item: item.date)
        if not sorted_daily:
            _mark_missing(f"{code}:daily", missing, "empty")
        latest_daily_date = sorted_daily[-1].date if sorted_daily else None
        trading_dates = list(
            dict.fromkeys(bar.date for bar in sorted_daily if bar.date < end_date)
        )
        news_dates = list(
            dict.fromkeys(
                [end_date, *reversed(trading_dates[-self.news_lookback_trading_days :])]
            )
        )
        news_items = []
        for search_date in news_dates:
            if search_date is None:
                continue
            items = self._read_optional(
                f"{code}:news",
                lambda search_date=search_date: self.gateway.get_stock_news(
                    code, search_date
                ),
                missing,
                failure_detail=search_date,
            )
            news_items.extend(items or [])
        unique_news = _unique_news(
            [
                replace(item, published_at=_to_kst(item.published_at))
                for item in news_items
            ]
        )
        if not unique_news:
            _mark_missing(f"{code}:news", missing, "empty")
        sorted_closes = [bar.close_price for bar in sorted_daily]
        sorted_volumes = [bar.volume for bar in sorted_daily]
        current_price = quote.current_price
        atr14_percent = calculate_atr14_percent(sorted_daily, current_price)
        if atr14_percent is None:
            _mark_missing(
                f"{code}:atr14",
                missing,
                _atr_missing_reason(sorted_daily, current_price),
            )
        realized_volatility = calculate_realized_volatility_20d_percent(sorted_closes)
        if realized_volatility is None:
            _mark_missing(
                f"{code}:realized_volatility_20d",
                missing,
                _volatility_missing_reason(sorted_closes),
            )
        high_20d_distance, low_20d_distance = calculate_20d_price_distances(
            sorted_daily, current_price
        )
        if high_20d_distance is None:
            _mark_missing(
                f"{code}:high_20d_distance",
                missing,
                _range_missing_reason(sorted_daily, current_price, PRICE_RANGE_WINDOW),
            )
        if low_20d_distance is None:
            _mark_missing(
                f"{code}:low_20d_distance",
                missing,
                _range_missing_reason(sorted_daily, current_price, PRICE_RANGE_WINDOW),
            )
        high_52w_position, high_52w_days = calculate_52w_position(
            sorted_daily, current_price
        )
        if high_52w_position is None:
            _mark_missing(
                f"{code}:high_52w_position",
                missing,
                _range_missing_reason(
                    sorted_daily,
                    current_price,
                    WEEK_52_TRADING_DAYS,
                    allow_partial=True,
                ),
            )
        latest_ma20 = ma20.value if ma20 else None
        latest_ma60 = ma60.value if ma60 else None
        news = [
            NewsSnapshotItem(
                title=item.title,
                source=item.source,
                published_at=item.published_at,
                age_hours=_age_hours(created_at, item.published_at),
                after_close=(
                    item.published_at.timetz().replace(tzinfo=None) > MARKET_CLOSE_TIME
                ),
            )
            for item in unique_news[: self.news_limit]
        ]

        stock = StockSnapshot(
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
            atr14_percent=atr14_percent,
            realized_volatility_20d_percent=realized_volatility,
            high_20d_distance_percent=high_20d_distance,
            low_20d_distance_percent=low_20d_distance,
            high_52w_position_percent=high_52w_position,
            high_52w_period_trading_days=high_52w_days,
            news=news,
            industry=quote.industry,
        )
        return stock, latest_daily_date

    def _required_quote(self, code: str) -> StockQuote:
        """필수 현재가 실패에 종목과 원인을 붙여 전파한다."""

        try:
            return self.gateway.get_stock_quote(code)
        except MarketDataUnavailable as error:
            raise MarketDataUnavailable(
                f"{code}:quote({error.reason})", reason=error.reason
            ) from error

    def _build_market_context(
        self,
        market_index_start_date: str,
        market_start_date: str,
        end_date: str,
        missing: list[str],
    ) -> MarketContext:
        """시장 시계열별 최신값과 최근 국면 지표를 구성한다."""

        kospi_points = self._read_optional(
            "market:kospi",
            lambda: self.gateway.get_market_index(
                MarketIndex.KOSPI, market_index_start_date, end_date
            ),
            missing,
        )
        kosdaq_points = self._read_optional(
            "market:kosdaq",
            lambda: self.gateway.get_market_index(
                MarketIndex.KOSDAQ, market_index_start_date, end_date
            ),
            missing,
        )
        kospi_points = sorted(kospi_points or [], key=lambda point: point.date)
        kosdaq_points = sorted(kosdaq_points or [], key=lambda point: point.date)
        if not kospi_points:
            _mark_missing("market:kospi", missing, "empty")
        if not kosdaq_points:
            _mark_missing("market:kosdaq", missing, "empty")
        kospi_return_5d = calculate_index_return_percent(kospi_points, 5)
        kospi_return_20d = calculate_index_return_percent(kospi_points, 20)
        kospi_ma20_deviation = calculate_index_deviation_percent(kospi_points, 20)
        kospi_ma60_deviation = calculate_index_deviation_percent(kospi_points, 60)
        _record_market_metric_missing(
            "market:kospi_return_5d", kospi_return_5d, kospi_points, 6, missing
        )
        _record_market_metric_missing(
            "market:kospi_return_20d", kospi_return_20d, kospi_points, 21, missing
        )
        _record_market_metric_missing(
            "market:kospi_ma20_deviation",
            kospi_ma20_deviation,
            kospi_points,
            20,
            missing,
        )
        _record_market_metric_missing(
            "market:kospi_ma60_deviation",
            kospi_ma60_deviation,
            kospi_points,
            60,
            missing,
        )

        kosdaq_return_5d = calculate_index_return_percent(kosdaq_points, 5)
        kosdaq_return_20d = calculate_index_return_percent(kosdaq_points, 20)
        kosdaq_ma20_deviation = calculate_index_deviation_percent(kosdaq_points, 20)
        kosdaq_ma60_deviation = calculate_index_deviation_percent(kosdaq_points, 60)
        _record_market_metric_missing(
            "market:kosdaq_return_5d", kosdaq_return_5d, kosdaq_points, 6, missing
        )
        _record_market_metric_missing(
            "market:kosdaq_return_20d", kosdaq_return_20d, kosdaq_points, 21, missing
        )
        _record_market_metric_missing(
            "market:kosdaq_ma20_deviation",
            kosdaq_ma20_deviation,
            kosdaq_points,
            20,
            missing,
        )
        _record_market_metric_missing(
            "market:kosdaq_ma60_deviation",
            kosdaq_ma60_deviation,
            kosdaq_points,
            60,
            missing,
        )
        sp500_points = self._read_optional(
            "market:sp500",
            lambda: self.gateway.get_market_index(
                MarketIndex.SP500, market_start_date, end_date
            ),
            missing,
        )
        sp500_change = _latest_change_percent(sp500_points or [])
        if sp500_points is not None and sp500_change is None:
            _mark_missing("market:sp500", missing, "empty")

        vix = self._latest_optional(
            "market:vix",
            lambda: self.gateway.get_market_index(
                MarketIndex.VIX, market_start_date, end_date
            ),
            missing,
        )
        vkospi_points = self._read_optional(
            "market:vkospi",
            lambda: self.gateway.get_market_index(
                MarketIndex.VKOSPI, market_start_date, end_date
            ),
            missing,
        )
        vkospi_points = sorted(vkospi_points or [], key=lambda point: point.date)
        if not vkospi_points:
            _mark_missing("market:vkospi", missing, "empty")
        vkospi = vkospi_points[-1] if vkospi_points else None
        vkospi_change_5d_points = calculate_index_change_points(vkospi_points, 5)
        _record_market_metric_missing(
            "market:vkospi_change_5d_points",
            vkospi_change_5d_points,
            vkospi_points,
            6,
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
                MarketIndex.USD_KRW, market_start_date, end_date
            ),
            missing,
        )
        korea_3y = self._latest_optional(
            "market:korea_3y_yield",
            lambda: self.gateway.get_treasury_yield(
                TreasuryMaturity.KOREA_3Y, market_start_date, end_date
            ),
            missing,
        )
        us_10y = self._latest_optional(
            "market:us_10y_yield",
            lambda: self.gateway.get_treasury_yield(
                TreasuryMaturity.US_10Y, market_start_date, end_date
            ),
            missing,
        )
        return MarketContext(
            kospi_change_percent=_latest_change_percent(kospi_points),
            kosdaq_change_percent=_latest_change_percent(kosdaq_points),
            sp500_change_percent=sp500_change,
            volatility_index=vix.value if vix else None,
            vkospi=vkospi.value if vkospi else None,
            fear_greed_index=fear_greed.value if fear_greed else None,
            exchange_rate=exchange_rate.value if exchange_rate else None,
            korea_3y_yield=korea_3y.value if korea_3y else None,
            us_10y_yield=us_10y.value if us_10y else None,
            kospi_return_5d_percent=kospi_return_5d,
            kospi_return_20d_percent=kospi_return_20d,
            kospi_ma20_deviation_percent=kospi_ma20_deviation,
            kospi_ma60_deviation_percent=kospi_ma60_deviation,
            kosdaq_return_5d_percent=kosdaq_return_5d,
            kosdaq_return_20d_percent=kosdaq_return_20d,
            kosdaq_ma20_deviation_percent=kosdaq_ma20_deviation,
            kosdaq_ma60_deviation_percent=kosdaq_ma60_deviation,
            vkospi_change_5d_points=vkospi_change_5d_points,
        )

    @staticmethod
    def _read_optional[T](
        label: str,
        read: Callable[[], T],
        missing: list[str],
        failure_detail: str | None = None,
    ) -> T | None:
        """보조 조회 실패를 missing 항목으로 기록하고 계속한다."""

        try:
            return read()
        except MarketDataUnavailable as error:
            _mark_missing(label, missing, error.reason, failure_detail)
            return None

    @classmethod
    def _latest_optional[T: IndicatorPoint | IndexPoint](
        cls,
        label: str,
        read: Callable[[], Sequence[T]],
        missing: list[str],
    ) -> T | None:
        """시계열의 최신 값을 가져오고 오류나 빈 응답을 기록한다."""

        points = cls._read_optional(label, read, missing)
        if not points:
            _mark_missing(label, missing, "empty")
            return None
        return max(points, key=lambda point: point.date)


def _record_market_metric_missing(
    label: str,
    value: Decimal | None,
    points: Sequence[IndexPoint],
    required_points: int,
    missing: list[str],
) -> None:
    """계산할 시장 이력이 부족하거나 기준 값이 잘못되면 원인을 남긴다."""

    if value is not None or not points:
        return
    reason = (
        "insufficient_history" if len(points) < required_points else "invalid_baseline"
    )
    _mark_missing(label, missing, reason)


def _add_relative_strength_fields(
    stocks: Sequence[StockSnapshot],
    market_names: dict[str, str | None],
    market_context: MarketContext,
) -> list[StockSnapshot]:
    """완성된 종목 universe를 바탕으로 상대강도와 순위를 채운다."""

    market_returns = {
        MarketIndex.KOSPI: (
            market_context.kospi_return_5d_percent,
            market_context.kospi_return_20d_percent,
        ),
        MarketIndex.KOSDAQ: (
            market_context.kosdaq_return_5d_percent,
            market_context.kosdaq_return_20d_percent,
        ),
    }
    universe_returns = [stock.return_20d_percent for stock in stocks]
    enriched: list[StockSnapshot] = []
    for stock in stocks:
        market = market_index_for(market_names.get(stock.code))
        if market is None:
            benchmark_5d, benchmark_20d = None, None
        else:
            benchmark_5d, benchmark_20d = market_returns[market]
        relative_5d = _subtract_if_available(stock.return_5d_percent, benchmark_5d)
        relative_20d = _subtract_if_available(stock.return_20d_percent, benchmark_20d)
        peers = [
            (peer.industry, peer.return_20d_percent)
            for peer in stocks
            if peer.code != stock.code
        ]
        enriched.append(
            replace(
                stock,
                relative_strength_5d_percent=relative_5d,
                relative_strength_20d_percent=relative_20d,
                universe_rank_percentile_20d=calculate_universe_rank_percentile(
                    stock.return_20d_percent, universe_returns
                ),
                industry_relative_strength_20d_percent=(
                    calculate_industry_relative_strength_20d_percent(
                        stock.return_20d_percent, stock.industry, peers
                    )
                ),
            )
        )
    return enriched


def _subtract_if_available(
    value: Decimal | None, benchmark: Decimal | None
) -> Decimal | None:
    if value is None or benchmark is None:
        return None
    return value - benchmark


def _normalize_industry(industry: str | None) -> str | None:
    if industry is None:
        return None
    normalized = industry.strip().casefold()
    return normalized or None


def _unique_codes(codes: Sequence[str]) -> list[str]:
    """입력 순서를 보존하며 중복 종목 코드를 제거한다."""

    return list(dict.fromkeys(codes))


def _with_current_prices(
    account: AccountSnapshot, quotes: dict[str, StockQuote]
) -> AccountSnapshot:
    """현재가를 채우고 계좌 손익률에서 매입 단가를 계산한다."""

    positions = [
        replace(
            position,
            current_price=quotes[position.code].current_price,
            average_price=_average_purchase_price(
                position, quotes[position.code].current_price
            ),
        )
        for position in account.positions
    ]
    return AccountSnapshot(cash=account.cash, positions=positions)


def _average_purchase_price(
    position: Position, current_price: Decimal
) -> Decimal | None:
    """trading-api 손익률과 현재가에서 평균 매입가를 추정한다."""

    if position.average_price is not None:
        return position.average_price
    profit_loss_rate = position.profit_loss_rate
    if profit_loss_rate is None:
        return None
    return_factor = Decimal(1) + profit_loss_rate / Decimal(100)
    if return_factor <= 0:
        return None
    return current_price / return_factor


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


def _latest_daily_date(
    results: Sequence[tuple[StockSnapshot, str | None]],
) -> str | None:
    """스냅샷에 포함된 종목 일봉 중 가장 최신 날짜를 반환한다."""

    dates = [date for _, date in results if date is not None]
    return max(dates) if dates else None


def has_daily_request_failure(missing: Sequence[str]) -> bool:
    """empty 외 일봉 요청 실패가 있는지 확인한다."""

    return any(":daily(" in item and not item.endswith("(empty)") for item in missing)


def _market_session(
    now: datetime, as_of_trading_date: str | None, missing: Sequence[str]
) -> str:
    """현재 한국 시각과 일봉 날짜로 장 상태를 구분한다."""

    local_now = now.astimezone(MARKET_TIMEZONE)
    if local_now.weekday() >= 5:
        return "holiday"
    if local_now.time() < time(9):
        return "pre_open"
    today = local_now.date().strftime("%Y%m%d")
    if (
        as_of_trading_date is not None
        and as_of_trading_date < today
        and not has_daily_request_failure(missing)
    ):
        return "holiday"
    if local_now.time() < MARKET_CLOSE_TIME:
        return "open"
    return "closed"


def _unique_news(news: Sequence[NewsItem]) -> list[NewsItem]:
    """기사 key나 제목이 겹치면 최신 게시 기사를 남긴다."""

    unique = []
    seen_keys: set[str] = set()
    seen_titles: set[str] = set()
    for item in sorted(news, key=lambda value: value.published_at, reverse=True):
        title = item.title.casefold().strip()
        if (item.key and item.key in seen_keys) or (title and title in seen_titles):
            continue
        if item.key:
            seen_keys.add(item.key)
        if title:
            seen_titles.add(title)
        unique.append(item)
    return unique


def _to_kst(value: datetime) -> datetime:
    """timezone-naive 기사는 KST로 보고 모든 게시 시각을 KST로 정규화한다."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=MARKET_TIMEZONE)
    return value.astimezone(MARKET_TIMEZONE)


def _age_hours(created_at: datetime, published_at: datetime) -> float:
    """스냅샷 생성 시각과 게시 시각 사이의 경과 시간을 소수 첫째 자리로 계산한다."""

    seconds = Decimal(str((created_at - published_at).total_seconds()))
    return float(
        (seconds / Decimal(3600)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    )


def _data_basis(news_lookback_trading_days: int) -> DataBasis:
    """지표 계산에 사용한 코드 상수와 가격 기준을 스냅샷에 기록한다."""

    return DataBasis(
        price_basis="adjusted",
        rsi=RSI_WINDOW,
        rsi_signal_ema=RSI_SIGNAL_WINDOW,
        macd=(MACD_SHORT_WINDOW, MACD_LONG_WINDOW),
        macd_signal_ema=MACD_SIGNAL_WINDOW,
        ma=(MA_SHORT_WINDOW, MA_LONG_WINDOW),
        atr=ATR_WINDOW,
        realized_volatility=REALIZED_VOLATILITY_WINDOW,
        news_lookback_trading_days=news_lookback_trading_days,
    )


def _mark_missing(
    label: str,
    missing: list[str],
    reason: str = "unavailable",
    detail: str | None = None,
) -> None:
    """missing 목록에 같은 항목이 중복 추가되지 않도록 한다."""

    if detail is not None:
        entry = f"{label}({detail} {reason})"
        if entry not in missing:
            missing.append(entry)
        return
    if any(item == label or item.startswith(f"{label}(") for item in missing):
        return
    missing.append(f"{label}({reason})")
