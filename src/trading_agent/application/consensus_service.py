"""두 판단 포트를 병렬로 실행하고 fail-closed 합의 결과를 만든다."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

from trading_agent.application.errors import DecisionModelError
from trading_agent.application.ports import DecisionModel, ModelRun
from trading_agent.domain.consensus import ConsensusPolicy, ConsensusResult
from trading_agent.domain.models import (
    ConfidenceAggregation,
    ConsensusPolicyName,
    ModelDecision,
    TradeProposal,
)
from trading_agent.domain.recommendation import (
    ModelRecommendation,
    RecommendationConsensusPolicy,
    RecommendationConsensusResult,
)


@dataclass(frozen=True, slots=True)
class ConsensusDecisionResult:
    """두 모델 원문과 성공 시 합의 결과를 함께 보존한다."""

    claude_run: ModelRun[ModelDecision]
    codex_run: ModelRun[ModelDecision]
    consensus: ConsensusResult | None

    @property
    def failed(self) -> bool:
        """한 모델이라도 정상 판단을 내지 못했는지 반환한다."""

        return self.consensus is None

    @property
    def proposals(self) -> list[TradeProposal]:
        """실패 시 빈 목록, 성공 시 두 모델의 합의 제안만 반환한다."""

        return [] if self.consensus is None else list(self.consensus.proposals)


@dataclass(frozen=True, slots=True)
class RecommendationConsensusOutcome:
    """두 모델 추천 원문과 성공 시 합의 투자의견을 함께 보존한다."""

    claude_run: ModelRun[ModelRecommendation]
    codex_run: ModelRun[ModelRecommendation]
    consensus: RecommendationConsensusResult | None

    @property
    def failed(self) -> bool:
        """한 모델이라도 정상 응답을 내지 못했는지 반환한다."""

        return self.consensus is None


class ConsensusDecisionService:
    """같은 프롬프트를 두 모델에 전달해 일치 결과만 공개한다."""

    def __init__(
        self,
        claude: DecisionModel[ModelDecision],
        codex: DecisionModel[ModelDecision],
        policy: ConsensusPolicyName = ConsensusPolicyName.STRICT,
        confidence_aggregation: ConfidenceAggregation = ConfidenceAggregation.MIN,
    ) -> None:
        self._claude = claude
        self._codex = codex
        self._policy = ConsensusPolicyName(policy)
        self._confidence_aggregation = ConfidenceAggregation(confidence_aggregation)

    def decide(self, prompt: str) -> ConsensusDecisionResult:
        """두 호출을 모두 기다리고 하나라도 실패하면 제안을 비운다."""

        claude_run, codex_run = run_both(self._claude, self._codex, prompt)
        if not claude_run.succeeded or not codex_run.succeeded:
            return ConsensusDecisionResult(claude_run, codex_run, None)
        assert claude_run.decision is not None and codex_run.decision is not None
        consensus = ConsensusPolicy.reconcile(
            claude_run.decision,
            codex_run.decision,
            self._policy,
            self._confidence_aggregation,
        )
        return ConsensusDecisionResult(claude_run, codex_run, consensus)


class RecommendationConsensusService:
    """같은 추천 프롬프트를 두 모델에 전달해 일치한 투자의견만 공개한다."""

    def __init__(
        self,
        claude: DecisionModel[ModelRecommendation],
        codex: DecisionModel[ModelRecommendation],
        policy: ConsensusPolicyName = ConsensusPolicyName.STRICT,
        confidence_aggregation: ConfidenceAggregation = ConfidenceAggregation.MIN,
    ) -> None:
        self._claude = claude
        self._codex = codex
        self._policy = ConsensusPolicyName(policy)
        self._confidence_aggregation = ConfidenceAggregation(confidence_aggregation)

    def recommend(self, prompt: str) -> RecommendationConsensusOutcome:
        """두 호출을 모두 기다리고 하나라도 실패하면 추천을 비운다."""

        claude_run, codex_run = run_both(self._claude, self._codex, prompt)
        if not claude_run.succeeded or not codex_run.succeeded:
            return RecommendationConsensusOutcome(claude_run, codex_run, None)
        assert claude_run.decision is not None and codex_run.decision is not None
        consensus = RecommendationConsensusPolicy.reconcile(
            claude_run.decision,
            codex_run.decision,
            self._policy,
            self._confidence_aggregation,
        )
        return RecommendationConsensusOutcome(claude_run, codex_run, consensus)


def run_both[T](
    claude: DecisionModel[T], codex: DecisionModel[T], prompt: str
) -> tuple[ModelRun[T], ModelRun[T]]:
    """두 모델을 병렬로 실행하고 실패도 실행 기록으로 반환한다."""

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="decision-model") as pool:
        claude_future = pool.submit(_execute, claude, "claude", prompt)
        codex_future = pool.submit(_execute, codex, "codex", prompt)
        claude_run = claude_future.result()
        codex_run = codex_future.result()
    return claude_run, codex_run


def _execute[T](model: DecisionModel[T], model_id: str, prompt: str) -> ModelRun[T]:
    """모델 오류를 실행 메타데이터와 함께 반환 가능한 값으로 바꾼다."""

    started = perf_counter()
    try:
        return model.decide(prompt)
    except DecisionModelError as error:
        return error.run
    except Exception as error:
        return ModelRun(
            model_id=model_id,
            stdout="",
            stderr="",
            exit_code=None,
            duration_seconds=perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )
