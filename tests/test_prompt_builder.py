import json
from datetime import UTC, datetime
from decimal import Decimal

from trading_agent.application.prompt_builder import build_decision_prompt
from trading_agent.domain.models import (
    AccountSnapshot,
    MarketContext,
    MarketSnapshot,
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
            )
        ],
        market_context=MarketContext(volatility_index=None),
        missing=["market:vix", "news:005930"],
    )
    policy = RiskPolicy(
        max_position_weight=Decimal("0.2"),
        max_order_amount=Decimal("500000"),
        min_confidence=0.7,
        allowed_codes={"005930"},
        max_proposals_per_cycle=5,
    )

    prompt = build_decision_prompt(snapshot, policy)
    snapshot_json = prompt.split("스냅샷 JSON:\n", maxsplit=1)[1]
    serialized = json.loads(snapshot_json)

    assert serialized["account"]["cash"] == "1234.50"
    assert serialized["stocks"][0]["current_price"] == "70000.25"
    assert serialized["stocks"][0]["rsi"] is None
    assert serialized["market_context"]["volatility_index"] is None
    assert serialized["created_at"] == "2026-10-29T00:00:00+00:00"
    assert "장중에는 당일 미완성 누적 거래량" in prompt
    assert "직전 20일 평균" in prompt
    assert "news:005930" in prompt
    assert "500000" in prompt
    assert "보유하지 않은 종목은 SELL로 제안하지 마세요" in prompt
