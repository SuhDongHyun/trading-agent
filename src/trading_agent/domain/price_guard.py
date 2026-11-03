"""추천 손절·의견 철회 가격의 ATR 간격과 손익비 검사."""

from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import StrEnum

from trading_agent.domain.models import Action, ResearchSnapshot, to_decimal
from trading_agent.domain.recommendation import StockRecommendation


class PriceGuardMode(StrEnum):
    """가격 보정 정책의 실행 모드."""

    ADJUST = "adjust"
    WARN = "warn"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class PriceGuardPolicy:
    """추천 가격 가드의 순수 도메인 설정."""

    mode: PriceGuardMode = PriceGuardMode.ADJUST
    min_stop_atr_multiple: Decimal = Decimal("2.0")
    min_reward_risk: Decimal = Decimal("1.5")

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", PriceGuardMode(self.mode))
        multiple = to_decimal(self.min_stop_atr_multiple)
        reward_risk = to_decimal(self.min_reward_risk)
        if multiple <= 0:
            raise ValueError("min_stop_atr_multiple는 0보다 커야 합니다.")
        if reward_risk <= 0:
            raise ValueError("min_reward_risk는 0보다 커야 합니다.")
        object.__setattr__(self, "min_stop_atr_multiple", multiple)
        object.__setattr__(self, "min_reward_risk", reward_risk)


DEFAULT_PRICE_GUARD_POLICY = PriceGuardPolicy()


def apply_price_guard(
    recommendation: StockRecommendation,
    snapshot: ResearchSnapshot,
    policy: PriceGuardPolicy = DEFAULT_PRICE_GUARD_POLICY,
) -> StockRecommendation:
    """합의 가격을 ATR 폭으로 보정하고 최소 손익비 미달을 경고한다."""

    if policy.mode is PriceGuardMode.OFF:
        return recommendation
    stock = next(
        (item for item in snapshot.stocks if item.code == recommendation.code), None
    )
    current = stock.current_price if stock is not None else None
    atr_percent = stock.atr14_percent if stock is not None else None
    invalidation = recommendation.invalidation_price
    adjusted = False
    original = recommendation.original_invalidation_price
    warnings = list(recommendation.price_warnings)

    stop_level = None
    if (
        current is not None
        and current > 0
        and atr_percent is not None
        and atr_percent > 0
        and invalidation is not None
    ):
        atr_amount = current * atr_percent / Decimal(100)
        minimum_distance = atr_amount * policy.min_stop_atr_multiple
        if recommendation.rating is Action.BUY:
            stop_level = current - minimum_distance
            too_close = invalidation > stop_level
            if too_close and stop_level > 0:
                if policy.mode is PriceGuardMode.ADJUST:
                    invalidation = stop_level.to_integral_value(rounding=ROUND_FLOOR)
                    original = recommendation.invalidation_price
                    adjusted = True
                else:
                    _append_warning(warnings, _stop_warning(policy))
        elif recommendation.rating is Action.SELL:
            stop_level = current + minimum_distance
            too_close = invalidation < stop_level
            if too_close:
                if policy.mode is PriceGuardMode.ADJUST:
                    invalidation = stop_level.to_integral_value(rounding=ROUND_CEILING)
                    original = recommendation.invalidation_price
                    adjusted = True
                else:
                    _append_warning(warnings, _stop_warning(policy))

    reward_risk = _reward_risk_ratio(recommendation, current, invalidation)
    if reward_risk is not None and reward_risk < policy.min_reward_risk:
        _append_warning(
            warnings,
            f"목표/손절 손익비 {reward_risk:.2f}가 최소 "
            f"{policy.min_reward_risk:g}보다 낮습니다.",
        )

    return replace(
        recommendation,
        invalidation_price=invalidation,
        original_invalidation_price=original,
        price_adjusted=recommendation.price_adjusted or adjusted,
        price_warnings=warnings,
    )


def _reward_risk_ratio(
    recommendation: StockRecommendation,
    current: Decimal | None,
    invalidation: Decimal | None,
) -> Decimal | None:
    if current is None or current <= 0 or invalidation is None:
        return None
    target = recommendation.target_price
    if target is None:
        return None
    if recommendation.rating is Action.BUY:
        reward = target - current
        risk = current - invalidation
    elif recommendation.rating is Action.SELL:
        reward = current - target
        risk = invalidation - current
    else:
        return None
    if risk <= 0:
        return Decimal(0)
    return reward / risk


def _stop_warning(policy: PriceGuardPolicy) -> str:
    return f"손절 폭이 최소 {policy.min_stop_atr_multiple:g}×ATR보다 좁습니다."


def _append_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)
