"""계좌와 무관한 종목 투자의견과 두 모델 합의 로직."""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from trading_agent.domain.consensus import (
    Disagreement,
    find_disagreement,
    group_by_code,
)
from trading_agent.domain.errors import DomainError
from trading_agent.domain.models import (
    Action,
    require_action,
    require_confidence,
    require_stock_code,
    to_decimal,
)


@dataclass(frozen=True, slots=True)
class StockRecommendation:
    """한 종목에 대한 투자의견, 목표가, 손절가."""

    code: str
    rating: Action
    confidence: float
    target_price: Decimal | None
    stop_loss_price: Decimal | None
    reason: str

    def __post_init__(self) -> None:
        require_stock_code(self.code)
        object.__setattr__(self, "rating", require_action(self.rating))
        object.__setattr__(self, "confidence", require_confidence(self.confidence))
        for field_name in ("target_price", "stop_loss_price"):
            value = getattr(self, field_name)
            if value is None:
                continue
            price = to_decimal(value)
            if price <= 0:
                raise DomainError("목표가와 손절가는 0보다 커야 합니다.")
            object.__setattr__(self, field_name, price)


@dataclass(frozen=True, slots=True)
class ModelRecommendation:
    """한 모델의 종목별 투자의견과 시장 판단 요약."""

    model_id: str
    recommendations: list[StockRecommendation]
    summary: str
    missing_info: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "recommendations", list(self.recommendations))
        object.__setattr__(self, "missing_info", list(self.missing_info))


@dataclass(frozen=True, slots=True)
class RecommendationConsensusResult:
    """두 모델이 같은 투자의견을 낸 종목과 불일치 기록."""

    recommendations: list[StockRecommendation]
    disagreements: list[Disagreement]
    missing_info: list[str]
    claude_summary: str
    codex_summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "recommendations", list(self.recommendations))
        object.__setattr__(self, "disagreements", list(self.disagreements))
        object.__setattr__(self, "missing_info", list(self.missing_info))

    def ranked(self) -> list[StockRecommendation]:
        """BUY, HOLD, SELL 순서로, 같은 의견 안에서는 확신도 높은 순서로 정렬한다."""

        order = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}
        return sorted(
            self.recommendations,
            key=lambda item: (order[item.rating], -item.confidence),
        )


class RecommendationConsensusPolicy:
    """두 모델의 투자의견이 같은 종목만 보수적인 가격으로 병합한다.

    매매 판단과 같은 규칙으로, 응답에서 빠진 종목은 HOLD 의견으로 간주한다.
    """

    @staticmethod
    def reconcile(
        claude: ModelRecommendation, codex: ModelRecommendation
    ) -> RecommendationConsensusResult:
        """종목 순서를 보존해 합의 의견과 불일치를 만든다."""

        claude_by_code = group_by_code(claude.recommendations)
        codex_by_code = group_by_code(codex.recommendations)
        codes = dict.fromkeys([*claude_by_code, *codex_by_code])
        agreed: list[StockRecommendation] = []
        disagreements: list[Disagreement] = []

        for code in codes:
            claude_items = claude_by_code.get(code, [])
            codex_items = codex_by_code.get(code, [])
            disagreement = find_disagreement(
                code,
                tuple(item.rating for item in claude_items),
                tuple(item.rating for item in codex_items),
            )
            if disagreement is not None:
                disagreements.append(disagreement)
            elif claude_items and codex_items:
                agreed.append(_merge(claude_items[0], codex_items[0]))

        return RecommendationConsensusResult(
            recommendations=agreed,
            disagreements=disagreements,
            missing_info=list(
                dict.fromkeys([*claude.missing_info, *codex.missing_info])
            ),
            claude_summary=claude.summary,
            codex_summary=codex.summary,
        )


def _merge(
    claude: StockRecommendation, codex: StockRecommendation
) -> StockRecommendation:
    """같은 의견을 낮은 확신도와 보수적인 목표가·손절가로 합친다."""

    if claude.rating is Action.BUY:
        target = _both(claude.target_price, codex.target_price, min)
        stop = _both(claude.stop_loss_price, codex.stop_loss_price, max)
    elif claude.rating is Action.SELL:
        target = _both(claude.target_price, codex.target_price, max)
        stop = _both(claude.stop_loss_price, codex.stop_loss_price, min)
    else:
        target = stop = None
    return StockRecommendation(
        code=claude.code,
        rating=claude.rating,
        confidence=min(claude.confidence, codex.confidence),
        target_price=target,
        stop_loss_price=stop,
        reason=f"[claude] {claude.reason} / [codex] {codex.reason}",
    )


def _both(
    first: Decimal | None,
    second: Decimal | None,
    pick: Callable[[Decimal, Decimal], Decimal],
) -> Decimal | None:
    """두 값이 모두 있을 때만 보수적인 쪽을 고른다."""

    if first is None or second is None:
        return None
    return pick(first, second)
