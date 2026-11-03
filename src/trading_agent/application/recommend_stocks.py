"""계좌와 무관하게 후보 종목의 합의 투자의견을 만든다."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime

from trading_agent.application.consensus_service import (
    RecommendationConsensusOutcome,
    RecommendationConsensusService,
)
from trading_agent.application.errors import MarketDataUnavailable
from trading_agent.application.recommendation_ledger import RecommendationLedger
from trading_agent.application.recommendation_prompt import (
    build_recommendation_prompt,
)
from trading_agent.application.records import CycleStatus, RecommendationRunResult
from trading_agent.application.snapshot_builder import SnapshotBuilder
from trading_agent.domain.models import ResearchSnapshot
from trading_agent.domain.price_guard import (
    DEFAULT_PRICE_GUARD_POLICY,
    PriceGuardPolicy,
    apply_price_guard,
)
from trading_agent.domain.recommendation import StockRecommendation


class RecommendStocks:
    """후보 종목 스냅샷 → 두 모델 투자의견 → 합의 → 기록을 실행한다."""

    def __init__(
        self,
        snapshot_builder: SnapshotBuilder,
        consensus_service: RecommendationConsensusService,
        universe: Sequence[str],
        ledger: RecommendationLedger,
        clock: Callable[[], datetime],
        prompt_builder: Callable[[ResearchSnapshot], str] = (
            build_recommendation_prompt
        ),
        price_guard_policy: PriceGuardPolicy = DEFAULT_PRICE_GUARD_POLICY,
        top_candidates: int = 5,
    ) -> None:
        if not universe:
            raise ValueError("추천 후보 종목이 비어 있습니다.")
        self.snapshot_builder = snapshot_builder
        self.consensus_service = consensus_service
        self.universe = tuple(universe)
        self.ledger = ledger
        self.clock = clock
        self.prompt_builder = prompt_builder
        self.price_guard_policy = price_guard_policy
        if top_candidates <= 0:
            raise ValueError("top_candidates는 0보다 커야 합니다.")
        self.top_candidates_count = top_candidates

    def run(self) -> RecommendationRunResult:
        """추천 결과를 반환하고 성공·실패와 무관하게 기록을 남긴다."""

        run_id = self.ledger.start_recommendation(self.clock())
        snapshot = None
        prompt = None
        outcome = None
        try:
            snapshot = self.snapshot_builder.build_research(self.universe)
            prompt = self.prompt_builder(snapshot)
            outcome = self.consensus_service.recommend(prompt)
            result = _result_from_outcome(
                run_id,
                prompt,
                snapshot,
                outcome,
                self.price_guard_policy,
                self.top_candidates_count,
            )
        except MarketDataUnavailable as error:
            result = RecommendationRunResult(
                run_id=run_id,
                status=CycleStatus.DATA_FAILED,
                error=f"{type(error).__name__}: {error}",
                prompt=prompt,
                snapshot=snapshot,
                outcome=outcome,
            )
        except Exception as error:
            result = RecommendationRunResult(
                run_id=run_id,
                status=CycleStatus.ERROR,
                error=f"{type(error).__name__}: {error}",
                prompt=prompt,
                snapshot=snapshot,
                outcome=outcome,
            )
        except BaseException as error:
            interrupted = RecommendationRunResult(
                run_id=run_id,
                status=CycleStatus.ERROR,
                error=f"{type(error).__name__}: {error}",
            )
            try:
                self.ledger.finish_recommendation(interrupted, self.clock())
            except BaseException:
                pass
            raise error

        try:
            self.ledger.finish_recommendation(result, self.clock())
        except BaseException as error:
            fallback = RecommendationRunResult(
                run_id=run_id,
                status=CycleStatus.ERROR,
                error=f"{type(error).__name__}: {error}",
            )
            try:
                self.ledger.finish_recommendation(fallback, self.clock())
            except BaseException as fallback_error:
                raise error from fallback_error
            if isinstance(error, Exception):
                return fallback
            raise error
        return result


def _result_from_outcome(
    run_id: int,
    prompt: str,
    snapshot: ResearchSnapshot,
    outcome: RecommendationConsensusOutcome,
    price_guard_policy: PriceGuardPolicy = DEFAULT_PRICE_GUARD_POLICY,
    top_candidates: int = 5,
) -> RecommendationRunResult:
    """모델 실패는 model_failed로, 성공은 스냅샷 종목만 순위대로 담는다."""

    if outcome.consensus is None:
        errors = [
            f"{run.model_id}: {run.error}"
            for run in (outcome.claude_run, outcome.codex_run)
            if run.error
        ]
        return RecommendationRunResult(
            run_id=run_id,
            status=CycleStatus.MODEL_FAILED,
            error="; ".join(errors) or "두 모델 합의 추천에 실패했습니다.",
            prompt=prompt,
            snapshot=snapshot,
            outcome=outcome,
        )
    snapshot_codes = {stock.code for stock in snapshot.stocks}
    consensus = replace(
        outcome.consensus,
        recommendations=[
            apply_price_guard(item, snapshot, price_guard_policy)
            for item in _within_snapshot(outcome.consensus.recommendations, snapshot)
        ],
        disagreements=[
            item
            for item in outcome.consensus.disagreements
            if item.code in snapshot_codes
        ],
        candidates=[
            item for item in outcome.consensus.candidates if item.code in snapshot_codes
        ],
    )
    outcome = replace(outcome, consensus=consensus)
    return RecommendationRunResult(
        run_id=run_id,
        status=CycleStatus.COMPLETED,
        prompt=prompt,
        snapshot=snapshot,
        outcome=outcome,
        recommendations=consensus.ranked(),
        top_candidates=consensus.top_candidates(top_candidates),
    )


def _within_snapshot(
    recommendations: list[StockRecommendation], snapshot: ResearchSnapshot
) -> list[StockRecommendation]:
    """스냅샷에 없는 종목에 대한 의견은 근거 데이터가 없으므로 버린다."""

    codes = {stock.code for stock in snapshot.stocks}
    return [item for item in recommendations if item.code in codes]
