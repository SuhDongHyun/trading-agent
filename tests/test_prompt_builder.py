import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_agent.application.prompt_builder import (
    CHASE_BUY_RETURN_5D_THRESHOLD_PERCENT,
    DEFAULT_HORIZON_DAYS,
    OVERHEAT_RSI_THRESHOLD,
    TRADE_DECISION_GUIDANCE,
    build_decision_prompt,
    format_horizon_period,
    to_json_value,
)
from trading_agent.domain.models import (
    AccountSnapshot,
    DataBasis,
    MarketContext,
    MarketSnapshot,
    NewsSnapshotItem,
    Position,
    StockSnapshot,
)
from trading_agent.domain.risk import RiskPolicy


def test_prompt_serializes_snapshot_values_and_includes_policy_glossary() -> None:
    snapshot = MarketSnapshot(
        created_at=datetime(2026, 10, 29, tzinfo=UTC),
        account=AccountSnapshot(
            cash=Decimal("1234.50"),
            positions=[
                Position("005930", "삼성전자", 2, current_price=Decimal("70000"))
            ],
        ),
        stocks=[
            StockSnapshot(
                "005930",
                "삼성전자",
                current_price=Decimal("70000.25"),
                volume_change_percent=Decimal("-12.5"),
                atr14_percent=Decimal("1.234"),
                realized_volatility_20d_percent=Decimal("2.345"),
                news=[
                    NewsSnapshotItem(
                        "장 마감 뒤 기사",
                        "연합뉴스",
                        datetime.fromisoformat("2026-10-29T15:31:00+09:00"),
                        0.2,
                        True,
                    )
                ],
            )
        ],
        market_context=MarketContext(volatility_index=None),
        missing=["market:vix", "news:005930"],
        data_basis=DataBasis(
            price_basis="adjusted",
            rsi=14,
            rsi_signal_ema=9,
            macd=(12, 26),
            macd_signal_ema=9,
            ma=(20, 60),
            atr=14,
            realized_volatility=20,
            news_lookback_trading_days=3,
        ),
    )
    policy = RiskPolicy(
        max_position_weight=Decimal("0.2"),
        max_order_amount=Decimal("500000"),
        min_confidence=0.7,
        allowed_codes={"005930"},
        max_proposals_per_cycle=5,
    )

    prompt = build_decision_prompt(snapshot, policy)
    snapshot_json = prompt.split("스냅샷 JSON:\n", maxsplit=1)[1].rstrip("\n")
    serialized = json.loads(snapshot_json)
    assert "\n" not in snapshot_json
    assert snapshot_json == json.dumps(
        serialized,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert serialized == json.loads(
        json.dumps(
            to_json_value(snapshot), ensure_ascii=False, indent=2, allow_nan=False
        )
    )

    risk_policy_json = prompt.split("리스크 정책:\n", maxsplit=1)[1].split(
        "\n\n필드 설명:", maxsplit=1
    )[0]
    serialized_policy = json.loads(risk_policy_json)
    expected_policy = {
        "max_position_weight": str(policy.max_position_weight),
        "max_order_ratio": str(policy.max_order_ratio),
        "max_order_amount": str(policy.max_order_amount),
        "effective_max_order_amount": str(
            policy.effective_order_limit(snapshot.account.total_asset_value)
        ),
        "stop_loss_percent": None,
        "take_profit_percent": None,
        "take_profit_sell_fraction": str(policy.take_profit_sell_fraction),
        "min_confidence": policy.min_confidence,
        "allowed_codes": sorted(policy.allowed_codes),
        "max_proposals_per_cycle": policy.max_proposals_per_cycle,
    }
    assert "\n" not in risk_policy_json
    assert risk_policy_json == json.dumps(
        serialized_policy,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert serialized_policy == json.loads(
        json.dumps(expected_policy, ensure_ascii=False, indent=2, allow_nan=False)
    )

    assert serialized["account"]["cash"] == "1235"
    assert serialized["stocks"][0]["current_price"] == "70000"
    assert serialized["stocks"][0]["rsi"] is None
    assert serialized["market_context"]["volatility_index"] is None
    assert "index_summary" not in serialized["market_context"]
    assert serialized["as_of_trading_date"] is None
    assert serialized["market_session"] == "closed"
    assert serialized["created_at"] == "2026-10-29T00:00:00+00:00"
    assert snapshot.account.cash == Decimal("1234.50")
    assert snapshot.stocks[0].current_price == Decimal("70000.25")
    assert serialized["stocks"][0]["volume_change_percent"] == "-12.50"
    assert serialized["stocks"][0]["atr14_percent"] == "1.23"
    assert serialized["stocks"][0]["realized_volatility_20d_percent"] == "2.35"
    assert serialized["stocks"][0]["news"] == [
        {
            "title": "장 마감 뒤 기사",
            "source": "연합뉴스",
            "published_at": "2026-10-29T15:31:00+09:00",
            "age_hours": 0.2,
            "after_close": True,
        }
    ]
    assert serialized["data_basis"] == {
        "price_basis": "adjusted",
        "rsi": 14,
        "rsi_signal_ema": 9,
        "macd": [12, 26],
        "macd_signal_ema": 9,
        "ma": [20, 60],
        "atr": 14,
        "realized_volatility": 20,
        "news_lookback_trading_days": 3,
    }
    assert "Wilder ATR(14)" in prompt
    assert "연율화하지 않았습니다" in prompt
    assert "장중에는 당일 미완성 누적 거래량" in prompt
    assert "직전 20일 평균" in prompt
    assert "stocks[].news" in prompt
    assert "after_close는" in prompt
    assert "15:30 KST 뒤 게시된" in prompt
    assert "data_basis.price_basis" in prompt
    assert "stocks[].news_headlines" not in prompt
    assert "news:005930" in prompt
    assert "500000" in prompt
    assert "max_order_ratio" in prompt
    assert "effective_max_order_amount" in prompt
    assert "손절·익절·비중 초과 축소는 시스템 규칙이 자동 처리" in prompt
    assert "보유하지 않은 종목은 SELL로 제안하지 마세요" in prompt
    assert "모든 종목에 대해 proposals 항목을 정확히 하나씩" in prompt
    assert "action을 hold, quantity를 0" in prompt
    assert (
        "snapshot.missing에 이미 있는 항목은 missing_info에 다시 적지 마세요" in prompt
    )
    assert "as_of_trading_date" in prompt
    assert "market_session" in prompt
    assert "0.5는 근거가 엇갈려 동전 던지기 수준" in prompt
    assert "0.7은 같은 근거로 10번 중 7번 맞을 수준" in prompt
    assert "hold의 confidence도 매수·매도하지 않기로 한 판단 자체" in prompt
    assert "10~20거래일(약 2~4주)" in prompt
    assert TRADE_DECISION_GUIDANCE in prompt
    assert f"stocks[].rsi가 {OVERHEAT_RSI_THRESHOLD} 이상" in prompt
    assert "stocks[].macd가 stocks[].macd_signal_value를 하향 교차" in prompt
    assert "stocks[].macd_signal이" in prompt
    assert "stocks[].relative_strength_5d_percent" in prompt
    assert "종목 소속 시장 지수(KOSPI 또는 KOSDAQ)" in prompt
    assert "market_name" not in prompt
    assert "kospi_ma60_deviation_percent" in prompt
    assert "vkospi_change_5d_points" in prompt
    assert "지수가 60일선 위에 있고 20일 수익률이 양수면 추세장" in prompt
    assert "종가의 20일선 이탈" in " ".join(prompt.split())
    assert "거래량 증가 동반 하락" in prompt
    assert (
        f"stocks[].return_5d_percent가 "
        f"{CHASE_BUY_RETURN_5D_THRESHOLD_PERCENT}% 이상" in prompt
    )
    assert "매매 근거가 약하거나 판단 확신이 낮으면" in prompt
    configured_prompt = build_decision_prompt(snapshot, policy, horizon_days=[5, 15])
    assert "5~15거래일(약 1~3주)" in configured_prompt


def test_recommendation_prompt_asks_for_every_stock_without_account() -> None:
    from trading_agent.application.recommendation_prompt import (
        build_recommendation_prompt,
    )
    from trading_agent.domain.models import ResearchSnapshot

    snapshot = ResearchSnapshot(
        created_at=datetime(2026, 11, 2, tzinfo=UTC),
        stocks=[
            StockSnapshot(
                code="086520",
                name="에코프로",
                atr14_percent=Decimal("3.456"),
                high_20d_distance_percent=Decimal("-4.5"),
                high_52w_position_percent=Decimal("75"),
                high_52w_period_trading_days=249,
            )
        ],
        market_context=MarketContext(),
        missing=["market:vkospi"],
        market_session="open",
    )

    prompt = build_recommendation_prompt(snapshot)
    serialized = json.loads(prompt.split("스냅샷 JSON:\n", maxsplit=1)[1])

    assert "account" not in serialized
    assert serialized["stocks"][0]["code"] == "086520"
    assert serialized["stocks"][0]["atr14_percent"] == "3.46"
    assert serialized["stocks"][0]["high_20d_distance_percent"] == "-4.50"
    assert serialized["stocks"][0]["high_52w_period_trading_days"] == 249
    assert "recommendations 항목을 정확히 하나씩" in prompt
    assert (
        "score와 expected_excess_return_10d로 후보끼리 상대 순위를 반드시 매기세요"
        in prompt
    )
    assert "score는 후보 universe 안의 상대 매력도이며 -2~+2" in prompt
    assert "drivers는 아래 스냅샷 근거로 확인되는 항목만 최대 3개" in prompt
    assert "key_risk는 주요 위험 하나를 type과 note로 작성하세요" in prompt
    assert "이미 있는 항목은 다시 적지 마세요" in prompt
    assert "market:vkospi" in prompt
    assert "stocks[].industry" in prompt
    assert "stocks[].industry_relative_strength_20d_percent" in prompt
    assert "현재가 × atr14_percent ÷ 100" in prompt
    assert "현재가 ÷ (1 + low_20d_distance_percent ÷ 100)" in prompt
    assert "현재가 ÷ (1 + high_20d_distance_percent ÷ 100)" in prompt
    assert "2×ATR" in prompt
    assert "최근 20거래일 최저 저가" in prompt
    assert "손익비는 최소 1.5" in prompt
    assert "market_session이 open이면 최신 일봉에 당일 미완성 봉" in prompt
    assert "as_of_trading_date" in prompt
    assert "0.5는 근거가 엇갈려 동전 던지기 수준" in prompt
    assert "hold의 confidence도 매수·매도하지 않기로 한 판단 자체" in prompt
    assert "10~20거래일(약 2~4주)" in prompt
    assert TRADE_DECISION_GUIDANCE in prompt
    assert f"stocks[].rsi가 {OVERHEAT_RSI_THRESHOLD} 이상" in prompt
    assert "stocks[].macd가 stocks[].macd_signal_value를 하향 교차" in prompt
    assert "stocks[].macd_signal이" in prompt
    assert "종가의 20일선 이탈" in " ".join(prompt.split())
    assert "거래량 증가 동반 하락" in prompt
    assert (
        f"stocks[].return_5d_percent가 "
        f"{CHASE_BUY_RETURN_5D_THRESHOLD_PERCENT}% 이상" in prompt
    )
    configured_prompt = build_recommendation_prompt(snapshot, horizon_days=[10])
    assert "10거래일 스윙" in configured_prompt


def test_horizon_period_uses_trading_day_range() -> None:
    assert DEFAULT_HORIZON_DAYS == (10, 20)
    assert format_horizon_period([10, 20]) == "10~20거래일(약 2~4주)"
    assert format_horizon_period([10]) == "10거래일"


@pytest.mark.parametrize("horizon_days", [[], [0], [20, 10], [10, 10], [True]])
def test_horizon_period_rejects_invalid_values(horizon_days: list[object]) -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        format_horizon_period(horizon_days)
