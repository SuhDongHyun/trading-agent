"""두 모델의 판단을 합의 제안으로 병합하는 순수 로직."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from trading_agent.domain.models import Action, ModelDecision, TradeProposal


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


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """두 모델의 합의 제안과 함께 보존할 판단 요약."""

    proposals: list[TradeProposal]
    disagreements: list[Disagreement]
    missing_info: list[str]
    claude_summary: str
    codex_summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposals", list(self.proposals))
        object.__setattr__(self, "disagreements", list(self.disagreements))
        object.__setattr__(self, "missing_info", list(self.missing_info))


class ConsensusPolicy:
    """두 모델의 일치하는 매수·매도 제안만 안전하게 병합한다.

    응답에서 빠진 종목은 HOLD로 간주하므로, HOLD와 미응답은 같은 의견이다.
    """

    @staticmethod
    def reconcile(claude: ModelDecision, codex: ModelDecision) -> ConsensusResult:
        """종목 순서와 missing 정보 순서를 보존해 합의 결과를 만든다."""

        claude_by_code = group_by_code(claude.proposals)
        codex_by_code = group_by_code(codex.proposals)
        codes = dict.fromkeys([*claude_by_code, *codex_by_code])
        proposals: list[TradeProposal] = []
        disagreements: list[Disagreement] = []

        for code in codes:
            claude_items = claude_by_code.get(code, [])
            codex_items = codex_by_code.get(code, [])
            disagreement = find_disagreement(
                code,
                tuple(item.action for item in claude_items),
                tuple(item.action for item in codex_items),
            )
            if disagreement is not None:
                disagreements.append(disagreement)
                continue
            if not claude_items or claude_items[0].action is Action.HOLD:
                continue
            proposals.append(_merge(claude_items[0], codex_items[0]))

        return ConsensusResult(
            proposals=proposals,
            disagreements=disagreements,
            missing_info=list(
                dict.fromkeys([*claude.missing_info, *codex.missing_info])
            ),
            claude_summary=claude.summary,
            codex_summary=codex.summary,
        )


def group_by_code[T: HasCode](items: list[T]) -> dict[str, list[T]]:
    """각 종목의 항목을 원래 순서대로 모은다."""

    grouped: dict[str, list[T]] = {}
    for proposal in items:
        grouped.setdefault(proposal.code, []).append(proposal)
    return grouped


def _merge(claude: TradeProposal, codex: TradeProposal) -> TradeProposal:
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
        confidence=min(claude.confidence, codex.confidence),
        reason=f"[claude] {claude.reason} / [codex] {codex.reason}",
    )
