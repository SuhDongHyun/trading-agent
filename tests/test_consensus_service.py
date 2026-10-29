from decimal import Decimal

import pytest

from trading_agent.application.consensus_service import ConsensusDecisionService
from trading_agent.application.errors import DecisionModelError
from trading_agent.application.ports import DecisionModel, ModelRun
from trading_agent.domain.models import Action, ModelDecision, TradeProposal


def make_decision(action: Action = Action.BUY) -> ModelDecision:
    proposal = TradeProposal(
        code="005930",
        action=action,
        quantity=2 if action is not Action.HOLD else 0,
        limit_price=Decimal("100"),
        confidence=0.8,
        reason="signal",
    )
    return ModelDecision("fake", [proposal], "summary", [])


class FakeDecisionModel(DecisionModel):
    def __init__(
        self,
        model_id: str,
        decision: ModelDecision | None = None,
        error: Exception | None = None,
    ) -> None:
        self.model_id = model_id
        self.decision = decision
        self.error = error
        self.prompts: list[str] = []

    def decide(self, prompt: str) -> ModelRun:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        assert self.decision is not None
        return ModelRun(
            model_id=self.model_id,
            stdout=f"{self.model_id} raw output",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            decision=self.decision,
        )


def failed_model(model_id: str) -> FakeDecisionModel:
    run = ModelRun(
        model_id=model_id,
        stdout=f"{model_id} partial output",
        stderr="failure detail",
        exit_code=1,
        duration_seconds=0.2,
        error="exit code 1",
    )
    return FakeDecisionModel(model_id, error=DecisionModelError("model failed", run))


def test_both_successful_models_return_only_matching_consensus() -> None:
    claude = FakeDecisionModel("claude", make_decision())
    codex = FakeDecisionModel("codex", make_decision())

    result = ConsensusDecisionService(claude, codex).decide("shared prompt")

    assert not result.failed
    assert result.consensus is not None
    assert len(result.proposals) == 1
    assert claude.prompts == ["shared prompt"]
    assert codex.prompts == ["shared prompt"]
    assert result.claude_run.stdout == "claude raw output"
    assert result.codex_run.stdout == "codex raw output"


def test_both_successful_but_disagreeing_models_return_no_proposal() -> None:
    claude = FakeDecisionModel("claude", make_decision(Action.BUY))
    codex = FakeDecisionModel("codex", make_decision(Action.SELL))

    result = ConsensusDecisionService(claude, codex).decide("shared prompt")

    assert not result.failed
    assert result.proposals == []
    assert result.consensus is not None
    assert len(result.consensus.disagreements) == 1


@pytest.mark.parametrize("failing_model", ["claude", "codex", "both"])
def test_any_model_failure_fails_closed_and_preserves_both_runs(
    failing_model: str,
) -> None:
    claude = (
        failed_model("claude")
        if failing_model in {"claude", "both"}
        else FakeDecisionModel("claude", make_decision())
    )
    codex = (
        failed_model("codex")
        if failing_model in {"codex", "both"}
        else FakeDecisionModel("codex", make_decision())
    )

    result = ConsensusDecisionService(claude, codex).decide("shared prompt")

    assert result.failed
    assert result.consensus is None
    assert result.proposals == []
    assert result.claude_run.model_id == "claude"
    assert result.codex_run.model_id == "codex"
    if failing_model == "claude":
        assert result.claude_run.stdout == "claude partial output"
        assert result.codex_run.succeeded
    elif failing_model == "codex":
        assert result.codex_run.stderr == "failure detail"
        assert result.claude_run.succeeded
    else:
        assert not result.claude_run.succeeded
        assert not result.codex_run.succeeded


def test_unexpected_model_exception_is_preserved_as_failure() -> None:
    claude = FakeDecisionModel("claude", error=RuntimeError("unexpected"))
    codex = FakeDecisionModel("codex", make_decision())

    result = ConsensusDecisionService(claude, codex).decide("prompt")

    assert result.failed
    assert result.proposals == []
    assert result.claude_run.error == "RuntimeError: unexpected"
