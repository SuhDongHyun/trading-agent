"""두 모델의 판단을 합의 제안으로 병합하는 순수 로직."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from trading_agent.domain.models import (
    Action,
    ConfidenceAggregation,
    ConsensusBasis,
    ConsensusPolicyName,
    ModelDecision,
    TradeProposal,
)


class HasCode(Protocol):
    """종목 코드로 묶을 수 있는 응답 항목."""

    @property
    def code(self) -> str: ...


class DisagreementReasonCode(StrEnum):
    """합의하지 못한 종목의 사유 코드."""

    ACTION_MISMATCH = "action_mismatch"
    DUPLICATE_PROPOSAL = "duplicate_proposal"


@dataclass(frozen=True, slots=True)
class Disagreement:
    """한 종목의 불일치 사유와 모델별 제안 동작."""

    code: str
    reason_code: DisagreementReasonCode
    claude_actions: tuple[Action, ...]
    codex_actions: tuple[Action, ...]


def effective_action(actions: tuple[Action, ...]) -> Action:
    """종목을 응답에서 뺀 모델은 HOLD로 간주한다."""

    return actions[0] if actions else Action.HOLD


def find_disagreement(
    code: str,
    claude_actions: tuple[Action, ...],
    codex_actions: tuple[Action, ...],
) -> Disagreement | None:
    """중복 응답이나 서로 다른 유효 동작이면 불일치를 반환한다."""

    if len(claude_actions) > 1 or len(codex_actions) > 1:
        reason = DisagreementReasonCode.DUPLICATE_PROPOSAL
    elif effective_action(claude_actions) is not effective_action(codex_actions):
        reason = DisagreementReasonCode.ACTION_MISMATCH
    else:
        return None
    return Disagreement(code, reason, claude_actions, codex_actions)


def resolve_consensus_actions(
    code: str,
    claude_actions: tuple[Action, ...],
    codex_actions: tuple[Action, ...],
    policy: ConsensusPolicyName = ConsensusPolicyName.STRICT,
) -> tuple[ConsensusBasis | None, Disagreement | None]:
    """정책에 따라 합의 항목의 근거 또는 불일치 사유를 정한다."""

    policy = ConsensusPolicyName(policy)
    disagreement = find_disagreement(code, claude_actions, codex_actions)
    if disagreement is None:
        return ConsensusBasis.AGREED, None

    if (
        policy is ConsensusPolicyName.BUY_ONE_SIDE
        and disagreement.reason_code is DisagreementReasonCode.ACTION_MISMATCH
    ):
        claude_action = effective_action(claude_actions)
        codex_action = effective_action(codex_actions)
        if claude_action is Action.BUY and codex_action is Action.HOLD:
            return ConsensusBasis.CLAUDE_ONLY, None
        if codex_action is Action.BUY and claude_action is Action.HOLD:
            return ConsensusBasis.CODEX_ONLY, None
    return None, disagreement


def aggregate_confidence(
    claude: float,
    codex: float,
    aggregation: ConfidenceAggregation,
) -> float:
    """두 확신도를 선택한 집계 규칙으로 합친다."""

    aggregation = ConfidenceAggregation(aggregation)
    if aggregation is ConfidenceAggregation.MIN:
        return min(claude, codex)
    mean = (Decimal(str(claude)) + Decimal(str(codex))) / Decimal("2")
    return float(mean)


def consensus_reason(
    claude_reason: str,
    codex_reason: str,
    basis: ConsensusBasis,
    *,
    claude_present: bool = True,
    codex_present: bool = True,
) -> str:
    """채택 근거와 HOLD 쪽 의견 또는 응답 누락을 보존한다."""

    claude_label = "[claude:hold]" if basis is ConsensusBasis.CODEX_ONLY else "[claude]"
    codex_label = "[codex:hold]" if basis is ConsensusBasis.CLAUDE_ONLY else "[codex]"
    claude_text = claude_reason if claude_present else "응답에 종목 없음"
    codex_text = codex_reason if codex_present else "응답에 종목 없음"
    return f"{claude_label} {claude_text} / {codex_label} {codex_text}"


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """두 모델의 합의 제안과 함께 보존할 판단 요약."""

    proposals: list[TradeProposal]
    disagreements: list[Disagreement]
    missing_info: list[str]
    claude_summary: str
    codex_summary: str
    policy: ConsensusPolicyName = ConsensusPolicyName.STRICT
    confidence_aggregation: ConfidenceAggregation = ConfidenceAggregation.MIN

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposals", list(self.proposals))
        object.__setattr__(self, "disagreements", list(self.disagreements))
        object.__setattr__(self, "missing_info", list(self.missing_info))


class ConsensusPolicy:
    """두 모델의 일치하는 매수·매도 제안만 안전하게 병합한다.

    응답에서 빠진 종목은 HOLD로 간주하므로, HOLD와 미응답은 같은 의견이다.
    """

    @staticmethod
    def reconcile(
        claude: ModelDecision,
        codex: ModelDecision,
        policy: ConsensusPolicyName = ConsensusPolicyName.STRICT,
        confidence_aggregation: ConfidenceAggregation = ConfidenceAggregation.MIN,
    ) -> ConsensusResult:
        """종목 순서와 missing 정보 순서를 보존해 합의 결과를 만든다."""

        policy = ConsensusPolicyName(policy)
        confidence_aggregation = ConfidenceAggregation(confidence_aggregation)

        claude_by_code = group_by_code(claude.proposals)
        codex_by_code = group_by_code(codex.proposals)
        codes = dict.fromkeys([*claude_by_code, *codex_by_code])
        proposals: list[TradeProposal] = []
        disagreements: list[Disagreement] = []

        for code in codes:
            claude_items = claude_by_code.get(code, [])
            codex_items = codex_by_code.get(code, [])
            basis, disagreement = resolve_consensus_actions(
                code,
                tuple(item.action for item in claude_items),
                tuple(item.action for item in codex_items),
                policy,
            )
            if disagreement is not None:
                disagreements.append(disagreement)
                continue
            if basis in (ConsensusBasis.CLAUDE_ONLY, ConsensusBasis.CODEX_ONLY):
                assert basis is not None
                claude_item = claude_items[0] if claude_items else None
                codex_item = codex_items[0] if codex_items else None
                buy_item = (
                    claude_item if basis is ConsensusBasis.CLAUDE_ONLY else codex_item
                )
                assert buy_item is not None
                proposals.append(
                    _adopt_one_side(
                        buy_item,
                        claude_item,
                        codex_item,
                        basis,
                    )
                )
                continue
            if not claude_items or claude_items[0].action is Action.HOLD:
                continue
            assert basis is not None
            proposals.append(
                _merge(claude_items[0], codex_items[0], confidence_aggregation, basis)
            )

        return ConsensusResult(
            proposals=proposals,
            disagreements=disagreements,
            missing_info=list(
                dict.fromkeys([*claude.missing_info, *codex.missing_info])
            ),
            claude_summary=claude.summary,
            codex_summary=codex.summary,
            policy=policy,
            confidence_aggregation=confidence_aggregation,
        )


def group_by_code[T: HasCode](items: list[T]) -> dict[str, list[T]]:
    """각 종목의 항목을 원래 순서대로 모은다."""

    grouped: dict[str, list[T]] = {}
    for proposal in items:
        grouped.setdefault(proposal.code, []).append(proposal)
    return grouped


def _merge(
    claude: TradeProposal,
    codex: TradeProposal,
    confidence_aggregation: ConfidenceAggregation,
    basis: ConsensusBasis,
) -> TradeProposal:
    """동일 동작 제안을 보수적인 수량·가격·확신도로 병합한다."""

    if claude.action is Action.BUY:
        limit_price = min(claude.limit_price, codex.limit_price)
    else:
        limit_price = max(claude.limit_price, codex.limit_price)
    return TradeProposal(
        code=claude.code,
        action=claude.action,
        quantity=min(claude.quantity, codex.quantity),
        limit_price=Decimal(limit_price),
        confidence=aggregate_confidence(
            claude.confidence, codex.confidence, confidence_aggregation
        ),
        reason=consensus_reason(claude.reason, codex.reason, basis),
        consensus_basis=basis,
    )


def _adopt_one_side(
    buy: TradeProposal,
    claude: TradeProposal | None,
    codex: TradeProposal | None,
    basis: ConsensusBasis,
) -> TradeProposal:
    """한쪽 BUY의 주문 값과 확신도를 그대로 채택한다."""

    return TradeProposal(
        code=buy.code,
        action=buy.action,
        quantity=buy.quantity,
        limit_price=buy.limit_price,
        confidence=buy.confidence,
        reason=consensus_reason(
            claude.reason if claude is not None else "",
            codex.reason if codex is not None else "",
            basis,
            claude_present=claude is not None,
            codex_present=codex is not None,
        ),
        consensus_basis=basis,
    )
