"""스냅샷부터 합의 판단과 리스크 검토까지 한 사이클을 실행한다."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, time
from zoneinfo import ZoneInfo

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
from trading_agent.application.snapshot_builder import (
    SnapshotBuilder,
    has_daily_request_failure,
)
from trading_agent.domain.models import MarketSnapshot
from trading_agent.domain.risk import RiskGuard, RiskPolicy
from trading_agent.domain.rules import generate_rule_proposals

MARKET_TIMEZONE = ZoneInfo("Asia/Seoul")


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
        force: bool = False,
    ) -> None:
        self.snapshot_builder = snapshot_builder
        self.consensus_service = consensus_service
        self.risk_policy = risk_policy
        self.watchlist = frozenset(watchlist)
        self.ledger = ledger
        self.clock = clock
        self.force = force
        self.prompt_builder = prompt_builder

    def run(self) -> DecisionCycleResult:
        """사이클 결과를 반환하고 완료 또는 실패 기록을 남긴다."""

        run_id = self.ledger.start_cycle(self.clock())
        snapshot = None
        prompt = None
        decision = None
        verdicts = []
        try:
            skip_reason = None if self.force else _skip_reason(None, self.clock())
            if skip_reason is None:
                snapshot = self.snapshot_builder.build()
                if not self.force:
                    skip_reason = _skip_reason(snapshot, snapshot.created_at)
            if skip_reason is not None:
                result = DecisionCycleResult(
                    run_id=run_id,
                    status=CycleStatus.SKIPPED,
                    error=skip_reason,
                    snapshot=snapshot,
                )
            else:
                assert snapshot is not None
                allowed_codes = self.watchlist | frozenset(
                    position.code for position in snapshot.account.positions
                )
                policy = replace(self.risk_policy, allowed_codes=allowed_codes)
                rule_proposals = generate_rule_proposals(snapshot.account, policy)
                prompt = self.prompt_builder(snapshot, policy)
                decision = self.consensus_service.decide(prompt)
                if decision.failed:
                    verdicts = RiskGuard.review(
                        rule_proposals,
                        snapshot.account,
                        policy,
                    )
                    result = DecisionCycleResult(
                        run_id=run_id,
                        status=CycleStatus.MODEL_FAILED,
                        error=_model_failure_detail(decision),
                        prompt=prompt,
                        snapshot=snapshot,
                        decision=decision,
                        verdicts=verdicts,
                    )
                else:
                    assert decision.consensus is not None
                    verdicts = RiskGuard.review(
                        [*rule_proposals, *decision.consensus.proposals],
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
        if result.status in (CycleStatus.COMPLETED, CycleStatus.SKIPPED):
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


def _skip_reason(snapshot: MarketSnapshot | None, now: datetime) -> str | None:
    """주말 또는 장 시작 후 오늘 일봉이 없으면 skip 사유를 반환한다."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("clock는 timezone-aware datetime을 반환해야 합니다.")
    local_now = now.astimezone(MARKET_TIMEZONE)
    if local_now.weekday() >= 5:
        return "주말 휴장일이므로 모델 판단을 생략했습니다."
    if snapshot is None:
        return None
    today = local_now.date().strftime("%Y%m%d")
    latest_date = snapshot.as_of_trading_date
    if (
        local_now.time() >= time(9)
        and latest_date is not None
        and latest_date < today
        and not has_daily_request_failure(snapshot.missing)
    ):
        return (
            f"오늘 일봉이 없고 최신 일봉은 {latest_date}여서 모델 판단을 생략했습니다."
        )
    return None
