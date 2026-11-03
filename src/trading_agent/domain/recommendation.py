"""계좌와 무관한 종목 투자의견과 두 모델 합의 로직."""

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from trading_agent.domain.consensus import (
    Disagreement,
    aggregate_confidence,
    consensus_reason,
    group_by_code,
    resolve_consensus_actions,
)
from trading_agent.domain.errors import DomainError
from trading_agent.domain.models import (
    Action,
    ConfidenceAggregation,
    ConsensusBasis,
    ConsensusPolicyName,
    require_action,
    require_confidence,
    require_stock_code,
    to_decimal,
)

RECOMMENDATION_DRIVERS = frozenset(
    {
        "trend_up",
        "trend_break",
        "overheated",
        "oversold_rebound",
        "volume_surge",
        "volume_dry",
        "breakout",
        "support_hold",
        "news_positive",
        "news_negative",
        "macro_risk",
        "relative_strength",
        "relative_weakness",
        "volatility_high",
    }
)
KEY_RISK_TYPES = frozenset(
    {
        "trend_reversal",
        "overheat_pullback",
        "event_risk",
        "macro_shock",
        "liquidity",
        "data_gap",
        "other",
    }
)


@dataclass(frozen=True, slots=True)
class RecommendationEvidence:
    """한 모델이 선택한 근거와 주요 위험."""

    drivers: list[str] = field(default_factory=list)
    key_risk_type: str | None = None
    key_risk_note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "drivers", list(self.drivers))


@dataclass(frozen=True, slots=True)
class StockRecommendation:
    """한 종목에 대한 투자의견, 목표가, 의견 무효화 가격."""

    code: str
    rating: Action
    confidence: float
    target_price: Decimal | None
    invalidation_price: Decimal | None
    reason: str
    consensus_basis: ConsensusBasis | None = None
    original_invalidation_price: Decimal | None = None
    price_adjusted: bool = False
    price_warnings: list[str] = field(default_factory=list)
    score: Decimal | None = None
    expected_excess_return_10d: Decimal | None = None
    drivers: list[str] = field(default_factory=list)
    key_risk_type: str | None = None
    key_risk_note: str | None = None
    evidence_by_model: dict[str, RecommendationEvidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_stock_code(self.code)
        object.__setattr__(self, "rating", require_action(self.rating))
        object.__setattr__(self, "confidence", require_confidence(self.confidence))
        if self.consensus_basis is not None:
            try:
                basis = ConsensusBasis(self.consensus_basis)
            except (TypeError, ValueError) as error:
                raise DomainError("합의 근거가 올바르지 않습니다.") from error
            object.__setattr__(self, "consensus_basis", basis)
        for field_name in (
            "target_price",
            "invalidation_price",
            "original_invalidation_price",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            price = to_decimal(value)
            if price <= 0:
                raise DomainError("목표가와 의견 무효화 가격은 0보다 커야 합니다.")
            object.__setattr__(self, field_name, price)
        object.__setattr__(self, "price_warnings", list(self.price_warnings))
        for field_name, lower, upper in (
            ("score", Decimal("-2"), Decimal("2")),
            (
                "expected_excess_return_10d",
                Decimal("-50"),
                Decimal("50"),
            ),
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            number = _recommendation_number(value, field_name)
            if number < lower or number > upper:
                raise DomainError(f"{field_name} 범위는 {lower}~{upper}입니다.")
            object.__setattr__(self, field_name, number)
        drivers = list(self.drivers)
        if any(not isinstance(driver, str) for driver in drivers):
            raise DomainError("drivers는 문자열 목록이어야 합니다.")
        if len(drivers) > 3 or len(drivers) != len(set(drivers)):
            raise DomainError("drivers는 중복 없이 최대 3개여야 합니다.")
        if any(driver not in RECOMMENDATION_DRIVERS for driver in drivers):
            raise DomainError("drivers에 지원하지 않는 값이 있습니다.")
        object.__setattr__(self, "drivers", drivers)
        if self.key_risk_type is not None:
            if (
                not isinstance(self.key_risk_type, str)
                or self.key_risk_type not in KEY_RISK_TYPES
            ):
                raise DomainError("key_risk.type 값이 올바르지 않습니다.")
            if not isinstance(self.key_risk_note, str):
                raise DomainError("key_risk.note는 문자열이어야 합니다.")
        elif self.key_risk_note is not None:
            raise DomainError("key_risk.type이 필요합니다.")
        object.__setattr__(self, "evidence_by_model", dict(self.evidence_by_model))


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
    policy: ConsensusPolicyName = ConsensusPolicyName.STRICT
    confidence_aggregation: ConfidenceAggregation = ConfidenceAggregation.MIN
    candidates: list["RecommendationCandidate"] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recommendations", list(self.recommendations))
        object.__setattr__(self, "disagreements", list(self.disagreements))
        object.__setattr__(self, "missing_info", list(self.missing_info))
        object.__setattr__(self, "candidates", list(self.candidates))

    def ranked(self) -> list[StockRecommendation]:
        """BUY, HOLD, SELL 순서로, 같은 의견 안에서는 확신도 높은 순서로 정렬한다."""

        order = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}
        return sorted(
            self.recommendations,
            key=lambda item: (order[item.rating], -item.confidence),
        )

    def top_candidates(self, limit: int = 5) -> list["RecommendationCandidate"]:
        """rating과 무관하게 score가 높은 후보를 반환한다."""

        if limit <= 0:
            raise ValueError("후보 수는 0보다 커야 합니다.")
        return list(self.candidates[:limit])


@dataclass(frozen=True, slots=True)
class RecommendationCandidate:
    """두 모델 점수를 합쳐 상대 순위에 사용하는 후보."""

    code: str
    score: Decimal
    expected_excess_return_10d: Decimal | None
    claude_score: Decimal | None
    codex_score: Decimal | None


class RecommendationConsensusPolicy:
    """두 모델의 투자의견이 같은 종목만 보수적인 가격으로 병합한다.

    매매 판단과 같은 규칙으로, 응답에서 빠진 종목은 HOLD 의견으로 간주한다.
    """

    @staticmethod
    def reconcile(
        claude: ModelRecommendation,
        codex: ModelRecommendation,
        policy: ConsensusPolicyName = ConsensusPolicyName.STRICT,
        confidence_aggregation: ConfidenceAggregation = ConfidenceAggregation.MIN,
    ) -> RecommendationConsensusResult:
        """종목 순서를 보존해 합의 의견과 불일치를 만든다."""

        policy = ConsensusPolicyName(policy)
        confidence_aggregation = ConfidenceAggregation(confidence_aggregation)

        claude_by_code = group_by_code(claude.recommendations)
        codex_by_code = group_by_code(codex.recommendations)
        codes = dict.fromkeys([*claude_by_code, *codex_by_code])
        agreed: list[StockRecommendation] = []
        disagreements: list[Disagreement] = []

        for code in codes:
            claude_items = claude_by_code.get(code, [])
            codex_items = codex_by_code.get(code, [])
            basis, disagreement = resolve_consensus_actions(
                code,
                tuple(item.rating for item in claude_items),
                tuple(item.rating for item in codex_items),
                policy,
            )
            if disagreement is not None:
                disagreements.append(disagreement)
            elif basis in (ConsensusBasis.CLAUDE_ONLY, ConsensusBasis.CODEX_ONLY):
                assert basis is not None
                claude_item = claude_items[0] if claude_items else None
                codex_item = codex_items[0] if codex_items else None
                buy_item = (
                    claude_item if basis is ConsensusBasis.CLAUDE_ONLY else codex_item
                )
                assert buy_item is not None
                agreed.append(_adopt_one_side(buy_item, claude_item, codex_item, basis))
            elif claude_items and codex_items:
                assert basis is not None
                agreed.append(
                    _merge(
                        claude_items[0], codex_items[0], confidence_aggregation, basis
                    )
                )

        return RecommendationConsensusResult(
            recommendations=agreed,
            disagreements=disagreements,
            missing_info=list(
                dict.fromkeys([*claude.missing_info, *codex.missing_info])
            ),
            claude_summary=claude.summary,
            codex_summary=codex.summary,
            policy=policy,
            confidence_aggregation=confidence_aggregation,
            candidates=_rank_candidates(claude_by_code, codex_by_code),
        )


def _merge(
    claude: StockRecommendation,
    codex: StockRecommendation,
    confidence_aggregation: ConfidenceAggregation,
    basis: ConsensusBasis | None,
) -> StockRecommendation:
    """같은 의견을 낮은 확신도와 보수적인 목표가·무효화 가격으로 합친다."""

    if claude.rating is Action.BUY:
        target = _both(claude.target_price, codex.target_price, min)
        invalidation = _both(claude.invalidation_price, codex.invalidation_price, max)
    elif claude.rating is Action.SELL:
        target = _both(claude.target_price, codex.target_price, max)
        invalidation = _both(claude.invalidation_price, codex.invalidation_price, min)
    else:
        target = invalidation = None
    return StockRecommendation(
        code=claude.code,
        rating=claude.rating,
        confidence=aggregate_confidence(
            claude.confidence, codex.confidence, confidence_aggregation
        ),
        target_price=target,
        invalidation_price=invalidation,
        reason=consensus_reason(
            claude.reason, codex.reason, basis or ConsensusBasis.AGREED
        ),
        consensus_basis=basis,
        score=_average_optional(claude.score, codex.score),
        expected_excess_return_10d=_average_optional(
            claude.expected_excess_return_10d, codex.expected_excess_return_10d
        ),
        evidence_by_model={
            "claude": _evidence(claude),
            "codex": _evidence(codex),
        },
    )


def _adopt_one_side(
    buy: StockRecommendation,
    claude: StockRecommendation | None,
    codex: StockRecommendation | None,
    basis: ConsensusBasis,
) -> StockRecommendation:
    """한쪽 BUY의 가격·확신도를 그대로 채택한다."""

    return StockRecommendation(
        code=buy.code,
        rating=buy.rating,
        confidence=buy.confidence,
        target_price=buy.target_price,
        invalidation_price=buy.invalidation_price,
        reason=consensus_reason(
            claude.reason if claude is not None else "",
            codex.reason if codex is not None else "",
            basis,
            claude_present=claude is not None,
            codex_present=codex is not None,
        ),
        consensus_basis=basis,
        score=_average_optional(
            claude.score if claude is not None else None,
            codex.score if codex is not None else None,
        ),
        expected_excess_return_10d=_average_optional(
            claude.expected_excess_return_10d if claude is not None else None,
            codex.expected_excess_return_10d if codex is not None else None,
        ),
        evidence_by_model={
            key: _evidence(item)
            for key, item in (("claude", claude), ("codex", codex))
            if item is not None
        },
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


def _recommendation_number(value: object, label: str) -> Decimal:
    """추천 점수 필드를 유한 Decimal로 정규화한다."""

    if isinstance(value, bool):
        raise DomainError(f"{label}는 유한한 숫자여야 합니다.")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as error:
        raise DomainError(f"{label}는 유한한 숫자여야 합니다.") from error
    if not number.is_finite():
        raise DomainError(f"{label}는 유한한 숫자여야 합니다.")
    return number


def _average_optional(first: Decimal | None, second: Decimal | None) -> Decimal | None:
    values = [value for value in (first, second) if value is not None]
    return sum(values, Decimal(0)) / len(values) if values else None


def _evidence(item: StockRecommendation) -> RecommendationEvidence:
    return RecommendationEvidence(item.drivers, item.key_risk_type, item.key_risk_note)


def _rank_candidates(
    claude: dict[str, list[StockRecommendation]],
    codex: dict[str, list[StockRecommendation]],
) -> list[RecommendationCandidate]:
    """중복 응답을 제외하고 두 모델의 점수 평균으로 후보를 정렬한다."""

    claude_unique = unique_recommendations_by_code(
        [item for items in claude.values() for item in items]
    )
    codex_unique = unique_recommendations_by_code(
        [item for items in codex.values() for item in items]
    )
    output: list[RecommendationCandidate] = []
    for code in dict.fromkeys([*claude, *codex]):
        left = claude_unique.get(code)
        right = codex_unique.get(code)
        claude_score = left.score if left is not None else None
        codex_score = right.score if right is not None else None
        score = _average_optional(claude_score, codex_score)
        if score is None:
            continue
        output.append(
            RecommendationCandidate(
                code=code,
                score=score,
                expected_excess_return_10d=_average_optional(
                    left.expected_excess_return_10d if left else None,
                    right.expected_excess_return_10d if right else None,
                ),
                claude_score=claude_score,
                codex_score=codex_score,
            )
        )
    return sorted(output, key=lambda item: (-item.score, item.code))


def unique_recommendations_by_code(
    recommendations: list[StockRecommendation],
) -> dict[str, StockRecommendation]:
    """종목당 한 번만 응답된 의견만 남기고 중복 코드는 제외한다."""

    grouped = group_by_code(recommendations)
    return {code: items[0] for code, items in grouped.items() if len(items) == 1}
