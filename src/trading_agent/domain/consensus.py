"""두 모델의 판단을 합의 제안으로 병합하는 순수 로직."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from trading_agent.domain.models import Action, ModelDecision, TradeProposal


class DisagreementReasonCode(StrEnum):
    """합의하지 못한 종목의 사유 코드."""

    ONE_SIDED = "one_sided"
    ACTION_MISMATCH = "action_mismatch"
    DUPLICATE_PROPOSAL = "duplicate_proposal"


@dataclass(frozen=True, slots=True)
class Disagreement:
    """한 종목의 불일치 사유와 모델별 제안 동작."""

    code: str
    reason_code: DisagreementReasonCode
    claude_actions: tuple[Action, ...]
    codex_actions: tuple[Action, ...]


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
    """두 모델의 일치하는 매수·매도 제안만 안전하게 병합한다."""

    @staticmethod
    def reconcile(claude: ModelDecision, codex: ModelDecision) -> ConsensusResult:
        """종목 순서와 missing 정보 순서를 보존해 합의 결과를 만든다."""

        claude_by_code = _group_proposals(claude.proposals)
        codex_by_code = _group_proposals(codex.proposals)
        codes = dict.fromkeys([*claude_by_code, *codex_by_code])
        proposals: list[TradeProposal] = []
        disagreements: list[Disagreement] = []

        for code in codes:
            claude_items = claude_by_code.get(code, [])
            codex_items = codex_by_code.get(code, [])
            claude_actions = tuple(item.action for item in claude_items)
            codex_actions = tuple(item.action for item in codex_items)

            if len(claude_items) > 1 or len(codex_items) > 1:
                disagreements.append(
                    Disagreement(
                        code,
                        DisagreementReasonCode.DUPLICATE_PROPOSAL,
                        claude_actions,
                        codex_actions,
                    )
                )
                continue

            if not claude_items or not codex_items:
                disagreements.append(
                    Disagreement(
                        code,
                        DisagreementReasonCode.ONE_SIDED,
                        claude_actions,
                        codex_actions,
                    )
                )
                continue

            claude_proposal = claude_items[0]
            codex_proposal = codex_items[0]
            if (
                claude_proposal.action is Action.HOLD
                and codex_proposal.action is Action.HOLD
            ):
                continue
            if claude_proposal.action is not codex_proposal.action:
                disagreements.append(
                    Disagreement(
                        code,
                        DisagreementReasonCode.ACTION_MISMATCH,
                        claude_actions,
                        codex_actions,
                    )
                )
                continue

            proposals.append(_merge(claude_proposal, codex_proposal))

        return ConsensusResult(
            proposals=proposals,
            disagreements=disagreements,
            missing_info=list(
                dict.fromkeys([*claude.missing_info, *codex.missing_info])
            ),
            claude_summary=claude.summary,
            codex_summary=codex.summary,
        )


def _group_proposals(proposals: list[TradeProposal]) -> dict[str, list[TradeProposal]]:
    """각 종목의 제안을 원래 순서대로 모은다."""

    grouped: dict[str, list[TradeProposal]] = {}
    for proposal in proposals:
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
