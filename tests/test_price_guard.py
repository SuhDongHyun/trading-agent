from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_agent.domain.models import (
    Action,
    MarketContext,
    ResearchSnapshot,
    StockSnapshot,
)
from trading_agent.domain.price_guard import (
    PriceGuardMode,
    PriceGuardPolicy,
    apply_price_guard,
)
from trading_agent.domain.recommendation import StockRecommendation


def snapshot(
    current_price: Decimal | None = Decimal("1000"),
    atr_percent: Decimal | None = Decimal("5"),
) -> ResearchSnapshot:
    return ResearchSnapshot(
        created_at=datetime(2026, 11, 2, tzinfo=UTC),
        stocks=[
            StockSnapshot(
                "005930",
                "삼성전자",
                current_price=current_price,
                atr14_percent=atr_percent,
            )
        ],
        market_context=MarketContext(),
        missing=[],
    )


def recommendation(rating: Action, invalidation: Decimal) -> StockRecommendation:
    target = Decimal("1200") if rating is Action.BUY else Decimal("800")
    return StockRecommendation("005930", rating, 0.8, target, invalidation, "근거")


@pytest.mark.parametrize(
    ("rating", "mode", "expected", "adjusted", "has_stop_warning"),
    [
        (Action.BUY, PriceGuardMode.ADJUST, Decimal("900"), True, False),
        (Action.BUY, PriceGuardMode.WARN, Decimal("950"), False, True),
        (Action.BUY, PriceGuardMode.OFF, Decimal("950"), False, False),
        (Action.SELL, PriceGuardMode.ADJUST, Decimal("1100"), True, False),
        (Action.SELL, PriceGuardMode.WARN, Decimal("1050"), False, True),
        (Action.SELL, PriceGuardMode.OFF, Decimal("1050"), False, False),
    ],
)
def test_price_guard_mode_and_action_matrix(
    rating: Action,
    mode: PriceGuardMode,
    expected: Decimal,
    adjusted: bool,
    has_stop_warning: bool,
) -> None:
    original = recommendation(
        rating, Decimal("950") if rating is Action.BUY else Decimal("1050")
    )

    guarded = apply_price_guard(
        original,
        snapshot(),
        PriceGuardPolicy(mode=mode),
    )

    assert guarded.invalidation_price == expected
    assert guarded.price_adjusted is adjusted
    assert guarded.original_invalidation_price == (
        original.invalidation_price if adjusted else None
    )
    assert (
        any("손절 폭" in warning for warning in guarded.price_warnings)
        is has_stop_warning
    )


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (Action.BUY, Decimal("900")),
        (Action.SELL, Decimal("1101")),
    ],
)
def test_adjustment_rounds_outward_to_integer_prices(
    rating: Action, expected: Decimal
) -> None:
    original = recommendation(
        rating,
        Decimal("950") if rating is Action.BUY else Decimal("1050"),
    )
    guarded = apply_price_guard(
        original,
        snapshot(Decimal("1000.1")),
        PriceGuardPolicy(),
    )

    assert guarded.invalidation_price == expected


def test_missing_atr_skips_stop_adjustment_without_warning() -> None:
    original = recommendation(Action.BUY, Decimal("950"))

    guarded = apply_price_guard(
        original,
        snapshot(atr_percent=None),
        PriceGuardPolicy(),
    )

    assert guarded.invalidation_price == Decimal("950")
    assert guarded.price_adjusted is False
    assert guarded.price_warnings == []


def test_missing_current_price_skips_guard_without_warning() -> None:
    original = recommendation(Action.BUY, Decimal("950"))

    guarded = apply_price_guard(
        original,
        snapshot(current_price=None),
        PriceGuardPolicy(),
    )

    assert guarded.invalidation_price == Decimal("950")
    assert guarded.price_adjusted is False
    assert guarded.price_warnings == []


def test_reward_risk_below_threshold_adds_warning_without_adjusting_price() -> None:
    original = StockRecommendation(
        "005930", Action.BUY, 0.8, Decimal("1050"), Decimal("950"), "근거"
    )

    guarded = apply_price_guard(
        original,
        snapshot(atr_percent=None),
        PriceGuardPolicy(mode=PriceGuardMode.WARN),
    )

    assert guarded.invalidation_price == Decimal("950")
    assert guarded.price_adjusted is False
    assert len(guarded.price_warnings) == 1
    assert "손익비 1.00" in guarded.price_warnings[0]
