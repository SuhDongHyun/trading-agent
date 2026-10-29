"""trading-api HTTP 클라이언트와 시장 데이터 어댑터."""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx

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
from trading_agent.domain.models import AccountSnapshot, Position
from trading_agent.infrastructure.config import TradingApiSettings

_VALID_SIGNALS = {"buy", "sell", "neutral"}


class TradingApiClient:
    """응답 상태와 JSON을 확인하는 작은 HTTP 클라이언트."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def request_json(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """HTTP 오류, 연결 오류, 잘못된 JSON을 애플리케이션 오류로 바꾼다."""

        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self._client.request(method, url, json=json_body)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            raise MarketDataUnavailable(
                f"trading-api 요청 실패: {method} {path} (HTTP {status})"
            ) from error
        except httpx.RequestError as error:
            raise MarketDataUnavailable(
                f"trading-api 연결 실패: {method} {path}"
            ) from error

        try:
            return response.json()
        except (ValueError, UnicodeDecodeError) as error:
            raise MarketDataUnavailable(
                f"trading-api JSON 응답이 올바르지 않습니다: {method} {path}"
            ) from error

    def close(self) -> None:
        """내부에서 만든 HTTP 클라이언트만 닫는다."""

        if self._owns_client:
            self._client.close()


class TradingApiMarketDataGateway(MarketDataGateway):
    """trading-api 응답을 애플리케이션 반환형과 Decimal로 변환한다."""

    def __init__(
        self,
        settings: TradingApiSettings,
        client: httpx.Client | None = None,
    ) -> None:
        self.client = TradingApiClient(
            settings.base_url,
            timeout_seconds=settings.timeout_seconds,
            client=client,
        )

    def close(self) -> None:
        """내부 HTTP 클라이언트의 리소스를 해제한다."""

        self.client.close()

    def get_account(self) -> AccountSnapshot:
        """계좌 응답의 현금과 보유 종목을 도메인 모델로 변환한다."""

        data = self.client.request_json("GET", "/account")

        def parse(payload: Any) -> AccountSnapshot:
            body = _mapping(payload)
            positions = [
                Position(
                    code=_string(item, "code"),
                    name=_string(item, "name"),
                    quantity=_integer(item, "quantity"),
                    profit_loss_rate=_decimal(item, "unrealized_return"),
                )
                for item in _list(body, "positions")
            ]
            accounts = _list(body, "accounts")
            if not accounts:
                raise ValueError("accounts 응답이 비어 있습니다.")
            cash = sum(
                (_decimal(_mapping(account), "cash_balance") for account in accounts),
                Decimal(0),
            )
            return AccountSnapshot(cash=cash, positions=positions)

        return _decode("GET /account", data, parse)

    def get_stock_quote(self, code: str) -> StockQuote:
        """현재가 응답에서 종목명과 가격, 등락률을 읽는다."""

        data = self.client.request_json(
            "POST", "/stock_quote", {"market": "J", "code": code}
        )

        def parse(payload: Any) -> StockQuote:
            body = _mapping(payload)
            current_price = _decimal(body, "current_price")
            if current_price <= 0:
                raise ValueError("현재가는 0보다 커야 합니다.")
            return StockQuote(
                code=_string(body, "code"),
                name=_string(body, "name"),
                current_price=current_price,
                change_percent=_decimal(body, "price_diff_rate"),
            )

        return _decode("POST /stock_quote", data, parse)

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        """일봉 응답을 종가와 거래량 형식으로 변환한다."""

        data = self._historical_request(
            "/stock_quote/daily", code, start_date, end_date
        )
        return _decode(
            "POST /stock_quote/daily",
            data,
            lambda payload: [
                DailyBar(
                    date=_string(_mapping(item), "date"),
                    close_price=_decimal(_mapping(item), "close_price"),
                    volume=_integer(_mapping(item), "accumulated_volume"),
                )
                for item in _sequence(payload)
            ],
        )

    def get_moving_average(
        self,
        code: str,
        start_date: str,
        end_date: str,
        window: int,
    ) -> list[IndicatorPoint]:
        """이동평균 시계열 응답을 지표 포인트로 변환한다."""

        data = self._historical_request(
            "/stock_quote/daily/moving-average",
            code,
            start_date,
            end_date,
            window=window,
        )
        return _decode(
            "POST /stock_quote/daily/moving-average",
            data,
            lambda payload: _indicator_points(payload, "moving_average"),
        )

    def get_rsi(
        self,
        code: str,
        start_date: str,
        end_date: str,
        window: int = 14,
    ) -> list[IndicatorPoint]:
        """RSI 시계열 응답을 지표 포인트로 변환한다."""

        data = self._historical_request(
            "/stock_quote/indicator/rsi",
            code,
            start_date,
            end_date,
            rsi_window=window,
        )
        return _decode(
            "POST /stock_quote/indicator/rsi",
            data,
            lambda payload: _indicator_points(payload, "rsi"),
        )

    def get_rsi_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        rsi_window: int = 14,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        """RSI signal EMA 시계열과 API 신호를 변환한다."""

        data = self._historical_request(
            "/stock_quote/indicator/rsi-signal",
            code,
            start_date,
            end_date,
            rsi_window=rsi_window,
            ema_window=ema_window,
        )
        return _decode(
            "POST /stock_quote/indicator/rsi-signal",
            data,
            lambda payload: _indicator_points(payload, "rsi_ema", with_signal=True),
        )

    def get_macd(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
    ) -> list[IndicatorPoint]:
        """MACD 시계열 응답을 지표 포인트로 변환한다."""

        data = self._historical_request(
            "/stock_quote/indicator/macd",
            code,
            start_date,
            end_date,
            ema_short_window=short_window,
            ema_long_window=long_window,
        )
        return _decode(
            "POST /stock_quote/indicator/macd",
            data,
            lambda payload: _indicator_points(payload, "macd"),
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
        """MACD signal EMA 시계열과 API 신호를 변환한다."""

        data = self._historical_request(
            "/stock_quote/indicator/macd-signal",
            code,
            start_date,
            end_date,
            ema_short_window=short_window,
            ema_long_window=long_window,
            ema_window=ema_window,
        )
        return _decode(
            "POST /stock_quote/indicator/macd-signal",
            data,
            lambda payload: _indicator_points(payload, "macd_ema", with_signal=True),
        )

    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
        """뉴스 응답을 제목, 출처, 게시 시각으로 변환한다."""

        data = self.client.request_json(
            "POST",
            "/stock_news",
            {"code": code, "search_date": search_date},
        )
        return _decode(
            "POST /stock_news",
            data,
            lambda payload: [
                NewsItem(
                    title=_string(_mapping(item), "title"),
                    source=_string(_mapping(item), "source"),
                    published_at=_datetime(_mapping(item), "published_at"),
                )
                for item in _sequence(payload)
            ],
        )

    def get_market_index(
        self,
        index: MarketIndex,
        start_date: str,
        end_date: str,
    ) -> list[IndexPoint]:
        """시장 지표 응답을 공통 시계열 형식으로 변환한다."""

        path = f"/market-index/{index.value}"
        if index is MarketIndex.VIX:
            data = self.client.request_json(
                "POST", path, {"start_date": start_date, "end_date": end_date}
            )
            value_field = "value"
            change_field = None
        else:
            data = self.client.request_json(
                "POST", path, {"start_date": start_date, "end_date": end_date}
            )
            value_field = "close_price"
            change_field = (
                "price_diff_rate"
                if index
                in (
                    MarketIndex.KOSPI,
                    MarketIndex.KOSDAQ,
                    MarketIndex.VKOSPI,
                )
                else None
            )

        return _decode(
            f"POST {path}",
            data,
            lambda payload: [
                IndexPoint(
                    date=_string(_mapping(item), "date"),
                    value=_decimal(_mapping(item), value_field),
                    change_percent=(
                        _decimal(_mapping(item), change_field)
                        if change_field is not None
                        else None
                    ),
                )
                for item in _sequence(payload)
            ],
        )

    def get_fear_and_greed_index(self) -> IndexPoint:
        """최신 공포탐욕 지수와 갱신 날짜를 변환한다."""

        data = self.client.request_json("GET", "/market-index/fear-and-greed-index")

        def parse(payload: Any) -> IndexPoint:
            body = _mapping(payload)
            updated_at = _datetime(body, "updated_at")
            return IndexPoint(
                date=updated_at.strftime("%Y%m%d"),
                value=_decimal(body, "value"),
            )

        return _decode("GET /market-index/fear-and-greed-index", data, parse)

    def get_treasury_yield(
        self,
        maturity: TreasuryMaturity,
        start_date: str,
        end_date: str,
    ) -> list[IndexPoint]:
        """국채 응답을 수익률 시계열로 변환한다."""

        path = f"/market-index/treasury-yield/{maturity.value}"
        data = self.client.request_json(
            "POST", path, {"start_date": start_date, "end_date": end_date}
        )

        def parse(payload: Any) -> list[IndexPoint]:
            values = []
            for item in _sequence(payload):
                body = _mapping(item)
                raw_rate = body.get("yield_rate", body.get("close_price"))
                if raw_rate is None:
                    raise ValueError("yield_rate 응답이 없습니다.")
                values.append(
                    IndexPoint(
                        date=_string(body, "date"),
                        value=_decimal_value(raw_rate),
                    )
                )
            return values

        return _decode(f"POST {path}", data, parse)

    def _historical_request(
        self,
        path: str,
        code: str,
        start_date: str,
        end_date: str,
        **parameters: int,
    ) -> Any:
        """주식 기간 요청의 공통 JSON 필드를 구성한다."""

        body: dict[str, Any] = {
            "market": "J",
            "code": code,
            "start_date": start_date,
            "end_date": end_date,
            "adjusted_price": True,
        }
        body.update(parameters)
        return self.client.request_json("POST", path, body)


def _decode[T](label: str, payload: Any, parser: Callable[[Any], T]) -> T:
    """응답 스키마 불일치를 같은 애플리케이션 오류로 통일한다."""

    try:
        return parser(payload)
    except MarketDataUnavailable:
        raise
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise MarketDataUnavailable(
            f"trading-api 응답 형식이 올바르지 않습니다: {label}"
        ) from error


def _mapping(value: Any) -> dict[str, Any]:
    """객체 응답인지 확인한다."""

    if not isinstance(value, dict):
        raise ValueError("객체 응답이 필요합니다.")
    return value


def _sequence(value: Any) -> list[Any]:
    """배열 응답인지 확인한다."""

    if not isinstance(value, list):
        raise ValueError("배열 응답이 필요합니다.")
    return value


def _list(body: dict[str, Any], field: str) -> list[Any]:
    """필수 배열 필드를 확인한다."""

    return _sequence(body[field])


def _string(body: dict[str, Any], field: str) -> str:
    """필수 문자열 필드를 확인한다."""

    value = body[field]
    if not isinstance(value, str):
        raise ValueError(f"{field}는 문자열이어야 합니다.")
    return value


def _integer(body: dict[str, Any], field: str) -> int:
    """필수 정수 필드를 확인한다."""

    value = body[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field}는 정수여야 합니다.")
    return value


def _decimal(body: dict[str, Any], field: str) -> Decimal:
    """필수 숫자 필드를 유한한 Decimal로 바꾼다."""

    return _decimal_value(body[field])


def _decimal_value(value: Any) -> Decimal:
    """JSON 숫자를 유한한 Decimal로 바꾼다."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError("숫자 응답이 필요합니다.")
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("숫자 응답은 유한해야 합니다.")
    return result


def _datetime(body: dict[str, Any], field: str) -> datetime:
    """ISO 형식 날짜 시각 필드를 datetime으로 바꾼다."""

    value = _string(body, field)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _indicator_points(
    payload: Any,
    value_field: str,
    with_signal: bool = False,
) -> list[IndicatorPoint]:
    """일자별 지표 응답을 공통 포인트로 바꾼다."""

    points = []
    for item in _sequence(payload):
        body = _mapping(item)
        signal = None
        if with_signal:
            signal = _string(body, "signal")
            if signal not in _VALID_SIGNALS:
                raise ValueError("지원하지 않는 지표 신호입니다.")
        points.append(
            IndicatorPoint(
                date=_string(body, "date"),
                value=_decimal(body, value_field),
                signal=signal,
            )
        )
    return points
