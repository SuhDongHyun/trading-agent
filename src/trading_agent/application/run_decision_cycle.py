"""스냅샷부터 합의 판단과 리스크 검토까지 한 사이클을 실행한다."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime

from trading_agent.application.consensus_service import (
    ConsensusDecisionResult,
    ConsensusDecisionService,
)
from trading_agent.application.decision_ledger import DecisionLedger
from trading_agent.application.errors import MarketDataUnavailable
from trading_agent.application.prompt_builder import build_decision_prompt
from trading_agent.application.records import (
    CycleStatus,
    DecisionCycleResult,
)
from trading_agent.application.snapshot_builder import SnapshotBuilder
from trading_agent.domain.models import MarketSnapshot
from trading_agent.domain.risk import RiskGuard, RiskPolicy


class RunDecisionCycle:
    """주문 없이 한 번의 수집·판단·리스크 검토를 실행한다."""

    def __init__(
        self,
        snapshot_builder: SnapshotBuilder,
        consensus_service: ConsensusDecisionService,
        risk_policy: RiskPolicy,
        watchlist: Sequence[str],
        ledger: DecisionLedger,
        clock: Callable[[], datetime],
        prompt_builder: Callable[[MarketSnapshot, RiskPolicy], str] = (
            build_decision_prompt
        ),
    ) -> None:
        self.snapshot_builder = snapshot_builder
        self.consensus_service = consensus_service
        self.risk_policy = risk_policy
        self.watchlist = frozenset(watchlist)
        self.ledger = ledger
        self.clock = clock
        self.prompt_builder = prompt_builder

    def run(self) -> DecisionCycleResult:
        """사이클 결과를 반환하고 완료 또는 실패 기록을 남긴다."""

        run_id = self.ledger.start_cycle(self.clock())
        snapshot = None
        prompt = None
        decision = None
        verdicts = []
        try:
            snapshot = self.snapshot_builder.build()
            allowed_codes = self.watchlist | frozenset(
                position.code for position in snapshot.account.positions
            )
            policy = replace(self.risk_policy, allowed_codes=allowed_codes)
            prompt = self.prompt_builder(snapshot, policy)
            decision = self.consensus_service.decide(prompt)
            if decision.failed:
                result = DecisionCycleResult(
                    run_id=run_id,
                    status=CycleStatus.MODEL_FAILED,
                    error=_model_failure_detail(decision),
                    prompt=prompt,
                    snapshot=snapshot,
                    decision=decision,
                )
            else:
                assert decision.consensus is not None
                verdicts = RiskGuard.review(
                    decision.consensus.proposals,
                    snapshot.account,
                    policy,
                )
                result = DecisionCycleResult(
                    run_id=run_id,
                    status=CycleStatus.COMPLETED,
                    prompt=prompt,
                    snapshot=snapshot,
                    decision=decision,
                    verdicts=verdicts,
                )
        except MarketDataUnavailable as error:
            result = DecisionCycleResult(
                run_id=run_id,
                status=CycleStatus.DATA_FAILED,
                error=f"{type(error).__name__}: {error}",
                prompt=prompt,
                snapshot=snapshot,
                decision=decision,
                verdicts=verdicts,
            )
        except Exception as error:
            result = DecisionCycleResult(
                run_id=run_id,
                status=CycleStatus.ERROR,
                error=f"{type(error).__name__}: {error}",
                prompt=prompt,
                snapshot=snapshot,
                decision=decision,
                verdicts=verdicts,
            )

        finished_at = self.clock()
        if result.status is CycleStatus.COMPLETED:
            self.ledger.complete_cycle(result, finished_at)
        else:
            self.ledger.fail_cycle(result, finished_at)
        return result


def _model_failure_detail(decision: ConsensusDecisionResult) -> str:
    """실패한 두 모델의 오류를 사이클 기록에 표시한다."""

    details = []
    for model_run in (decision.claude_run, decision.codex_run):
        if model_run.error:
            details.append(f"{model_run.model_id}: {model_run.error}")
    return "; ".join(details) or "두 모델 합의 판단에 실패했습니다."
