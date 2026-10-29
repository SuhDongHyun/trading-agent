from datetime import datetime
from decimal import Decimal

import httpx
import pytest

from trading_agent.application.errors import MarketDataUnavailable
from trading_agent.application.ports import MarketIndex, TreasuryMaturity
from trading_agent.infrastructure.config import TradingApiSettings
from trading_agent.infrastructure.trading_api import TradingApiMarketDataGateway


def test_gateway_uses_trading_api_contract_and_parses_decimal_values() -> None:
    requests: list[tuple[str, str, dict | None]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = request.read()
        json_body = httpx.Response(200, content=body).json() if body else None
        requests.append((request.method, request.url.path, json_body))

        responses = {
            ("GET", "/account"): {
                "positions": [
                    {
                        "code": "005930",
                        "name": "삼성전자",
                        "quantity": 2,
                        "unrealized_pnl": 30,
                        "unrealized_return": 1.5,
                    }
                ],
                "accounts": [
                    {"cash_balance": 1000.25, "total_pnl": 30, "total_return": 1.5}
                ],
            },
            ("POST", "/stock_quote"): {
                "market_name": "KOSPI",
                "code": "005930",
                "name": "삼성전자",
                "industry": "전자",
                "per": 10,
                "pbr": 1,
                "eps": 100,
                "bps": 1000,
                "open_price": 100,
                "current_price": 101.5,
                "previous_price": 100,
                "highest_price": 102,
                "lowest_price": 99,
                "upper_limit_price": 130,
                "lower_limit_price": 70,
                "current_volume": 10,
                "previous_volume": 9,
                "current_trading_value": 1015,
                "price_diff": 1.5,
                "price_diff_rate": 1.5,
            },
            ("POST", "/stock_quote/daily"): [
                {
                    "date": "20261028",
                    "open_price": 100,
                    "high_price": 103,
                    "low_price": 99,
                    "close_price": 101.5,
                    "accumulated_volume": 500,
                    "accumulated_trading_value": 100,
                    "price_diff": 1,
                    "price_diff_sign": "2",
                    "change_flag": "1",
                }
            ],
            ("POST", "/stock_quote/daily/moving-average"): [
                {"date": "20261028", "moving_average": 99.5}
            ],
            ("POST", "/stock_quote/indicator/rsi"): [{"date": "20261028", "rsi": 55.5}],
            ("POST", "/stock_quote/indicator/rsi-signal"): [
                {"date": "20261028", "rsi_ema": 54.5, "signal": "buy"}
            ],
            ("POST", "/stock_quote/indicator/macd"): [
                {"date": "20261028", "macd": 1.25}
            ],
            ("POST", "/stock_quote/indicator/macd-signal"): [
                {"date": "20261028", "macd_ema": 1.0, "signal": "neutral"}
            ],
            ("POST", "/stock_news"): [
                {
                    "title": "삼성전자 소식",
                    "source": "연합뉴스",
                    "published_at": "2026-10-28T09:30:00+09:00",
                }
            ],
            ("POST", "/market-index/kospi-index"): [
                {
                    "date": "20261028",
                    "open_price": 1,
                    "high_price": 2,
                    "low_price": 1,
                    "close_price": 2,
                    "price_diff": 0.1,
                    "price_diff_rate": 0.5,
                    "volume": 100,
                    "trading_value": 1000,
                }
            ],
            ("POST", "/market-index/vix-index"): [{"date": "20261028", "value": 16.25}],
            ("GET", "/market-index/fear-and-greed-index"): {
                "value": 62.5,
                "condition": "greed",
                "updated_at": "2026-10-28T15:00:00+09:00",
            },
            ("POST", "/market-index/treasury-yield/korea-3y"): [
                {"date": "20261028", "yield_rate": 3.25}
            ],
        }
        return httpx.Response(200, json=responses[(request.method, request.url.path)])

    client = httpx.Client(transport=httpx.MockTransport(handle))
    gateway = TradingApiMarketDataGateway(
        TradingApiSettings(base_url="http://trading-api", timeout_seconds=2),
        client=client,
    )
    try:
        account = gateway.get_account()
        quote = gateway.get_stock_quote("005930")
        bars = gateway.get_daily_bars("005930", "20260701", "20261029")
        averages = gateway.get_moving_average("005930", "20260701", "20261029", 20)
        rsi = gateway.get_rsi("005930", "20260701", "20261029")
        rsi_signal = gateway.get_rsi_signal("005930", "20260701", "20261029")
        macd = gateway.get_macd("005930", "20260701", "20261029")
        macd_signal = gateway.get_macd_signal("005930", "20260701", "20261029")
        news = gateway.get_stock_news("005930", "20261029")
        kospi = gateway.get_market_index(MarketIndex.KOSPI, "20260701", "20261029")
        vix = gateway.get_market_index(MarketIndex.VIX, "20260701", "20261029")
        fear_greed = gateway.get_fear_and_greed_index()
        yield_points = gateway.get_treasury_yield(
            TreasuryMaturity.KOREA_3Y, "20260701", "20261029"
        )
    finally:
        client.close()

    assert account.cash == Decimal("1000.25")
    assert account.positions[0].profit_loss_rate == Decimal("1.5")
    assert quote.current_price == Decimal("101.5")
    assert quote.change_percent == Decimal("1.5")
    assert bars[0].close_price == Decimal("101.5")
    assert bars[0].volume == 500
    assert averages[0].value == Decimal("99.5")
    assert rsi[0].value == Decimal("55.5")
    assert rsi_signal[0].signal == "buy"
    assert macd[0].value == Decimal("1.25")
    assert macd_signal[0].value == Decimal("1.0")
    assert news[0].published_at == datetime.fromisoformat("2026-10-28T09:30:00+09:00")
    assert kospi[0].change_percent == Decimal("0.5")
    assert vix[0].value == Decimal("16.25")
    assert fear_greed.value == Decimal("62.5")
    assert yield_points[0].value == Decimal("3.25")

    assert requests[0] == ("GET", "/account", None)
    assert requests[1] == ("POST", "/stock_quote", {"market": "J", "code": "005930"})
    assert requests[2] == (
        "POST",
        "/stock_quote/daily",
        {
            "market": "J",
            "code": "005930",
            "start_date": "20260701",
            "end_date": "20261029",
            "adjusted_price": True,
        },
    )
    assert requests[3][2]["window"] == 20
    assert requests[5][2]["rsi_window"] == 14
    assert requests[6][2]["ema_short_window"] == 12
    assert requests[8] == (
        "POST",
        "/stock_news",
        {"code": "005930", "search_date": "20261029"},
    )
    assert requests[9][1] == "/market-index/kospi-index"
    assert requests[9][2] == {"start_date": "20260701", "end_date": "20261029"}
    assert requests[10][1] == "/market-index/vix-index"
    assert requests[11] == ("GET", "/market-index/fear-and-greed-index", None)
    assert requests[12][1] == "/market-index/treasury-yield/korea-3y"


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(503, json={"detail": "unavailable"}),
        lambda request: httpx.Response(200, content=b"not-json"),
        lambda request: httpx.Response(200, json={"unexpected": "shape"}),
    ],
    ids=["http-error", "json-error", "schema-error"],
)
def test_gateway_converts_http_json_and_schema_errors(handler) -> None:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = TradingApiMarketDataGateway(
        TradingApiSettings(base_url="http://trading-api"), client=client
    )
    try:
        with pytest.raises(MarketDataUnavailable):
            gateway.get_stock_quote("005930")
    finally:
        client.close()


def test_gateway_converts_connection_errors() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    gateway = TradingApiMarketDataGateway(
        TradingApiSettings(base_url="http://trading-api"), client=client
    )
    try:
        with pytest.raises(MarketDataUnavailable, match="연결 실패"):
            gateway.get_stock_quote("005930")
    finally:
        client.close()
