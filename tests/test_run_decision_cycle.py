import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fakes import FakeGateway

from trading_agent.application.consensus_service import ConsensusDecisionService
from trading_agent.application.errors import DecisionModelError, MarketDataUnavailable
from trading_agent.application.ports import (
    DailyBar,
    DecisionModel,
    ModelRun,
    StockQuote,
)
from trading_agent.application.records import CycleStatus
from trading_agent.application.run_decision_cycle import RunDecisionCycle
from trading_agent.application.snapshot_builder import SnapshotBuilder
from trading_agent.domain.models import (
    AccountSnapshot,
    Action,
    ConfidenceAggregation,
    ConsensusBasis,
    ConsensusPolicyName,
    ModelDecision,
    Position,
    TradeProposal,
)
from trading_agent.domain.risk import RiskPolicy, RiskReasonCode
from trading_agent.infrastructure.sqlite_ledger import SQLiteDecisionLedger

FIXED_NOW = datetime(2026, 10, 28, 23, 0, tzinfo=UTC)


class FakeDecisionModel(DecisionModel):
    """고정 모델 결과를 반환하는 테스트 어댑터."""

    def __init__(
        self,
        model_id: str,
        decision: ModelDecision | None,
        fail: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_decision = decision
        self.fail = fail
        self.calls = 0

    def decide(self, prompt: str) -> ModelRun:
        self.calls += 1
        if self.fail:
            run = ModelRun(
                self.model_id,
                stdout=f"{self.model_id} raw partial",
                stderr="failed",
                exit_code=1,
                duration_seconds=0.25,
                error="CLI exit code 1",
            )
            raise DecisionModelError("model failed", run)
        assert self.model_decision is not None
        return ModelRun(
            model_id=self.model_id,
            stdout=f"{self.model_id} raw output",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            decision=self.model_decision,
        )


def make_decision(model_id: str, proposals: list[TradeProposal]) -> ModelDecision:
    return ModelDecision(model_id, proposals, f"{model_id} summary", ["news"])


def make_proposal(
    code: str,
    price: str = "100.123456789",
    *,
    confidence: float = 0.9,
    action: Action = Action.BUY,
) -> TradeProposal:
    return TradeProposal(
        code=code,
        action=action,
        quantity=0 if action is Action.HOLD else 2,
        limit_price=Decimal(price),
        confidence=confidence,
        reason="test signal",
    )


def build_cycle(
    db_path: Path,
    *,
    fail_account: bool = False,
    model_failure: bool = False,
    now: datetime = FIXED_NOW,
    gateway: FakeGateway | None = None,
    force: bool = False,
    clock_fn: Callable[[], datetime] | None = None,
) -> RunDecisionCycle:
    gateway = gateway or FakeGateway(fail_account)
    proposal = make_proposal("005930")
    claude = FakeDecisionModel(
        "claude",
        make_decision("claude", [proposal, make_proposal("000660")]),
        fail=model_failure,
    )
    codex = FakeDecisionModel(
        "codex",
        make_decision("codex", [make_proposal("005930", "100.123456788")]),
    )

    clock = clock_fn or (lambda: now)

    return RunDecisionCycle(
        snapshot_builder=SnapshotBuilder(gateway, ["005930"], clock=clock),
        consensus_service=ConsensusDecisionService(claude, codex),
        risk_policy=RiskPolicy(
            max_position_weight=Decimal("0.2"),
            max_order_amount=Decimal("1000"),
            min_confidence=0.7,
            allowed_codes=set(),
            max_proposals_per_cycle=5,
        ),
        watchlist=["005930"],
        ledger=SQLiteDecisionLedger(db_path),
        clock=clock,
        force=force,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_status"),
    [
        ({}, CycleStatus.COMPLETED),
        ({"model_failure": True}, CycleStatus.MODEL_FAILED),
        ({"fail_account": True}, CycleStatus.DATA_FAILED),
    ],
)
def test_cycle_records_completed_model_failed_and_data_failed(
    tmp_path: Path,
    kwargs: dict[str, bool],
    expected_status: CycleStatus,
) -> None:
    db_path = tmp_path / "nested" / "cycles.sqlite"
    result = build_cycle(
        db_path,
        fail_account=kwargs.get("fail_account", False),
        model_failure=kwargs.get("model_failure", False),
    ).run()

    assert result.status is expected_status
    assert db_path.is_file()
    import sqlite3

    with sqlite3.connect(db_path) as connection:
        status, finished_at, error = connection.execute(
            "SELECT status, finished_at, error FROM runs WHERE id = ?",
            (result.run_id,),
        ).fetchone()
        assert status == expected_status.value
        assert datetime.fromisoformat(finished_at).tzinfo is not None
        if expected_status is CycleStatus.COMPLETED:
            assert error is None
            assert (
                connection.execute(
                    """SELECT COUNT(*) FROM risk_verdicts
                       WHERE run_id = ? AND approved = 1""",
                    (result.run_id,),
                ).fetchone()[0]
                == 1
            )
            assert (
                connection.execute(
                    "SELECT limit_price FROM consensus_proposals WHERE run_id = ?",
                    (result.run_id,),
                ).fetchone()[0]
                == "100.123456788"
            )
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM consensus_disagreements WHERE run_id = ?",
                    (result.run_id,),
                ).fetchone()[0]
                == 1
            )
            assert (
                connection.execute(
                    "SELECT missing_info_json FROM consensus_results WHERE run_id = ?",
                    (result.run_id,),
                ).fetchone()[0]
                == '["news"]'
            )
        elif expected_status is CycleStatus.MODEL_FAILED:
            assert "claude" in error
            assert (
                connection.execute(
                    """SELECT stdout FROM model_runs
                   WHERE run_id = ? AND model_id = 'claude'""",
                    (result.run_id,),
                ).fetchone()[0]
                == "claude raw partial"
            )
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM risk_verdicts WHERE run_id = ?",
                    (result.run_id,),
                ).fetchone()[0]
                == 0
            )
        else:
            assert "account unavailable" in error
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM model_runs WHERE run_id = ?",
                    (result.run_id,),
                ).fetchone()[0]
                == 0
            )


def test_one_sided_buy_reaches_risk_guard_confidence_check(tmp_path: Path) -> None:
    db_path = tmp_path / "one-sided.sqlite"
    cycle = build_cycle(db_path)
    cycle.consensus_service = ConsensusDecisionService(
        FakeDecisionModel(
            "claude",
            make_decision("claude", [make_proposal("005930", confidence=0.59)]),
        ),
        FakeDecisionModel(
            "codex",
            make_decision("codex", [make_proposal("005930", action=Action.HOLD)]),
        ),
        policy=ConsensusPolicyName.BUY_ONE_SIDE,
        confidence_aggregation=ConfidenceAggregation.MEAN,
    )

    result = cycle.run()

    assert result.status is CycleStatus.COMPLETED
    assert result.decision is not None and result.decision.consensus is not None
    adopted = result.decision.consensus.proposals[0]
    assert adopted.consensus_basis is ConsensusBasis.CLAUDE_ONLY
    assert adopted.confidence == 0.59
    verdict = next(
        item for item in result.verdicts or [] if item.proposal.code == "005930"
    )
    assert not verdict.approved
    assert RiskReasonCode.CONFIDENCE_TOO_LOW in {
        reason.code for reason in verdict.reasons
    }

    with sqlite3.connect(db_path) as connection:
        policy, aggregation = connection.execute(
            "SELECT policy, confidence_aggregation FROM consensus_results "
            "WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()
        basis = connection.execute(
            "SELECT consensus_basis FROM consensus_proposals WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()[0]
    assert (policy, aggregation, basis) == ("buy_one_side", "mean", "claude_only")


def test_mean_confidence_at_minimum_threshold_is_approved(tmp_path: Path) -> None:
    cycle = build_cycle(tmp_path / "confidence-boundary.sqlite")
    cycle.risk_policy = replace(cycle.risk_policy, min_confidence=0.65)
    cycle.consensus_service = ConsensusDecisionService(
        FakeDecisionModel(
            "claude",
            make_decision("claude", [make_proposal("005930", confidence=0.6)]),
        ),
        FakeDecisionModel(
            "codex",
            make_decision("codex", [make_proposal("005930", confidence=0.7)]),
        ),
        policy=ConsensusPolicyName.STRICT,
        confidence_aggregation=ConfidenceAggregation.MEAN,
    )

    result = cycle.run()

    assert result.status is CycleStatus.COMPLETED
    assert result.decision is not None and result.decision.consensus is not None
    proposal = result.decision.consensus.proposals[0]
    assert proposal.confidence == 0.65
    verdict = next(
        item for item in result.verdicts or [] if item.proposal.code == "005930"
    )
    assert verdict.approved
    assert RiskReasonCode.CONFIDENCE_TOO_LOW not in {
        reason.code for reason in verdict.reasons
    }


def test_unexpected_exception_is_recorded_as_error(tmp_path: Path) -> None:
    db_path = tmp_path / "error.sqlite"
    cycle = build_cycle(db_path)

    def fail_prompt(snapshot, policy) -> str:
        raise RuntimeError("prompt failed")

    cycle.prompt_builder = fail_prompt
    result = cycle.run()

    assert result.status is CycleStatus.ERROR
    assert "RuntimeError: prompt failed" in (result.error or "")
    with sqlite3.connect(db_path) as connection:
        status, error = connection.execute(
            "SELECT status, error FROM runs WHERE id = ?", (result.run_id,)
        ).fetchone()
    assert status == "error"
    assert "RuntimeError: prompt failed" in error


def test_interruption_closes_run_as_error_then_reraises(tmp_path: Path) -> None:
    db_path = tmp_path / "interrupted.sqlite"
    cycle = build_cycle(db_path)

    class SimulatedInterrupt(BaseException):
        pass

    def interrupt(snapshot, policy) -> str:
        raise SimulatedInterrupt("stopped")

    cycle.prompt_builder = interrupt
    with pytest.raises(SimulatedInterrupt, match="stopped"):
        cycle.run()

    with sqlite3.connect(db_path) as connection:
        status, error = connection.execute("SELECT status, error FROM runs").fetchone()
    assert status == "error"
    assert "SimulatedInterrupt: stopped" in error


def test_recording_failure_falls_back_to_error_run(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "recording-error.sqlite"
    cycle = build_cycle(db_path)
    ledger = cycle.ledger

    def fail_first_complete(result, finished_at) -> None:
        raise RuntimeError("cycle write failed")

    monkeypatch.setattr(ledger, "complete_cycle", fail_first_complete)
    result = cycle.run()

    assert result.status is CycleStatus.ERROR
    assert "cycle write failed" in (result.error or "")
    with sqlite3.connect(db_path) as connection:
        status, error = connection.execute(
            "SELECT status, error FROM runs WHERE id = ?", (result.run_id,)
        ).fetchone()
    assert status == "error"
    assert "cycle write failed" in error


@pytest.mark.parametrize(
    ("now", "daily_date", "force", "expected_status", "expected_session"),
    [
        (
            datetime(2026, 11, 1, 10, tzinfo=ZoneInfo("Asia/Seoul")),
            None,
            False,
            CycleStatus.SKIPPED,
            "holiday",
        ),
        (
            datetime(2026, 11, 2, 8, 59, tzinfo=ZoneInfo("Asia/Seoul")),
            None,
            False,
            CycleStatus.COMPLETED,
            "pre_open",
        ),
        (
            datetime(2026, 11, 2, 9, tzinfo=ZoneInfo("Asia/Seoul")),
            None,
            False,
            CycleStatus.COMPLETED,
            "open",
        ),
        (
            datetime(2026, 11, 2, 10, tzinfo=ZoneInfo("Asia/Seoul")),
            "20261102",
            False,
            CycleStatus.COMPLETED,
            "open",
        ),
        (
            datetime(2026, 11, 2, 10, tzinfo=ZoneInfo("Asia/Seoul")),
            "20261030",
            False,
            CycleStatus.SKIPPED,
            "holiday",
        ),
        (
            datetime(2026, 11, 2, 10, tzinfo=ZoneInfo("Asia/Seoul")),
            "20261030",
            True,
            CycleStatus.COMPLETED,
            "holiday",
        ),
        (
            datetime(2026, 11, 1, 10, tzinfo=ZoneInfo("Asia/Seoul")),
            None,
            True,
            CycleStatus.COMPLETED,
            "holiday",
        ),
    ],
    ids=[
        "weekend",
        "pre-open",
        "open-empty-daily",
        "open-with-today-bar",
        "open-with-stale-bar",
        "force-stale-daily",
        "force-weekend",
    ],
)
def test_cycle_skips_without_llm_when_calendar_or_daily_data_requires_it(
    tmp_path: Path,
    now: datetime,
    daily_date: str | None,
    force: bool,
    expected_status: CycleStatus,
    expected_session: str,
) -> None:
    class DailyDateGateway(FakeGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            if daily_date is None:
                return []
            return [DailyBar(daily_date, Decimal("100"), 10)]

    cycle = build_cycle(
        tmp_path / "skip.sqlite",
        now=now,
        gateway=DailyDateGateway(),
        force=force,
    )

    result = cycle.run()

    assert result.status is expected_status
    if result.snapshot is None:
        assert expected_session == "holiday"
        assert expected_status is CycleStatus.SKIPPED
    else:
        assert result.snapshot.market_session == expected_session
    if expected_status is CycleStatus.SKIPPED:
        assert result.decision is None
        assert result.error is not None
    else:
        assert result.decision is not None
    with sqlite3.connect(tmp_path / "skip.sqlite") as connection:
        status, model_runs = connection.execute(
            "SELECT status, (SELECT COUNT(*) FROM model_runs WHERE run_id = runs.id) "
            "FROM runs WHERE id = ?",
            (result.run_id,),
        ).fetchone()
    assert status == expected_status.value
    assert model_runs == (0 if expected_status is CycleStatus.SKIPPED else 2)


def test_daily_request_failures_do_not_masquerade_as_a_holiday(
    tmp_path: Path,
) -> None:
    class FailedDailyGateway(FakeGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            raise MarketDataUnavailable("daily request timed out", reason="timeout")

    now = datetime(2026, 11, 2, 10, tzinfo=ZoneInfo("Asia/Seoul"))
    result = build_cycle(
        tmp_path / "daily-timeout.sqlite",
        now=now,
        gateway=FailedDailyGateway(),
    ).run()

    assert result.status is CycleStatus.COMPLETED
    assert result.snapshot is not None
    assert result.snapshot.as_of_trading_date is None
    assert result.snapshot.market_session == "open"
    assert "005930:daily(timeout)" in result.snapshot.missing
    assert "000660:daily(timeout)" in result.snapshot.missing
    assert result.decision is not None


def test_daily_failure_holds_stale_date_skip_and_session_classification(
    tmp_path: Path,
) -> None:
    class PartialDailyFailureGateway(FakeGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            if code == "005930":
                return [DailyBar("20261030", Decimal("100"), 10)]
            raise MarketDataUnavailable("daily request timed out", reason="timeout")

    now = datetime(2026, 11, 2, 10, tzinfo=ZoneInfo("Asia/Seoul"))
    result = build_cycle(
        tmp_path / "partial-daily-timeout.sqlite",
        now=now,
        gateway=PartialDailyFailureGateway(),
    ).run()

    assert result.status is CycleStatus.COMPLETED
    assert result.snapshot is not None
    assert result.snapshot.as_of_trading_date == "20261030"
    assert result.snapshot.market_session == "open"
    assert "000660:daily(timeout)" in result.snapshot.missing
    assert result.decision is not None


def test_skip_boundary_uses_snapshot_time_when_collection_crosses_open(
    tmp_path: Path,
) -> None:
    class StaleDailyGateway(FakeGateway):
        def get_daily_bars(
            self, code: str, start_date: str, end_date: str
        ) -> list[DailyBar]:
            return [DailyBar("20261030", Decimal("100"), 10)]

    times = iter(
        [
            datetime(2026, 11, 2, 8, 59, 50, tzinfo=ZoneInfo("Asia/Seoul")),
            datetime(2026, 11, 2, 8, 59, 50, tzinfo=ZoneInfo("Asia/Seoul")),
            datetime(2026, 11, 2, 8, 59, 59, tzinfo=ZoneInfo("Asia/Seoul")),
            datetime(2026, 11, 2, 9, 0, 20, tzinfo=ZoneInfo("Asia/Seoul")),
        ]
    )
    cycle = build_cycle(
        tmp_path / "boundary.sqlite",
        gateway=StaleDailyGateway(),
        clock_fn=lambda: next(times),
    )

    result = cycle.run()

    assert result.status is CycleStatus.COMPLETED
    assert result.snapshot is not None
    assert result.snapshot.created_at == datetime(
        2026, 11, 2, 8, 59, 59, tzinfo=ZoneInfo("Asia/Seoul")
    )


def test_prompt_rounding_does_not_change_precision_saved_in_snapshot_json(
    tmp_path: Path,
) -> None:
    class PreciseGateway(FakeGateway):
        def get_stock_quote(self, code: str) -> StockQuote:
            return StockQuote(
                code,
                "정밀 종목",
                Decimal("100.123456789"),
                Decimal("1.23456789"),
            )

    db_path = tmp_path / "precision.sqlite"
    result = build_cycle(db_path, gateway=PreciseGateway()).run()

    assert result.status is CycleStatus.COMPLETED
    assert result.prompt is not None
    prompt_snapshot = json.loads(result.prompt.split("스냅샷 JSON:\n", 1)[1])
    assert prompt_snapshot["stocks"][0]["current_price"] == "100"
    with sqlite3.connect(db_path) as connection:
        snapshot_json = connection.execute(
            "SELECT snapshot_json FROM runs WHERE id = ?", (result.run_id,)
        ).fetchone()[0]
    stored_snapshot = json.loads(snapshot_json)
    assert stored_snapshot["stocks"][0]["current_price"] == "100.123456789"


def test_sql_queries_find_latest_approvals_and_disagreement_history(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ledger.sqlite"
    result = build_cycle(db_path).run()

    import sqlite3

    with sqlite3.connect(db_path) as connection:
        latest_approved = connection.execute(
            """SELECT code, action, quantity, limit_price
               FROM risk_verdicts
               WHERE approved = 1 AND run_id = (
                   SELECT id FROM runs ORDER BY started_at DESC, id DESC LIMIT 1
               )"""
        ).fetchone()
        disagreement = connection.execute(
            """SELECT code, reason_code
               FROM consensus_disagreements WHERE run_id = ?""",
            (result.run_id,),
        ).fetchone()

    assert latest_approved == ("005930", "buy", 2, "100.123456788")
    assert disagreement == ("000660", "action_mismatch")


class StopLossGateway(FakeGateway):
    """현재가가 평균 매입가보다 낮은 단일 보유 계좌를 반환한다."""

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            Decimal("8500"),
            [Position("005930", "삼성전자", 15, profit_loss_rate=Decimal("-10"))],
        )

    def get_stock_quote(self, code: str) -> StockQuote:
        return StockQuote(code, "삼성전자", Decimal("100"), Decimal("0"))


def build_rule_cycle(db_path: Path, *, model_failure: bool = False) -> RunDecisionCycle:
    gateway = StopLossGateway()
    proposal = make_proposal("005930", "10")
    claude = FakeDecisionModel(
        "claude", make_decision("claude", [proposal]), fail=model_failure
    )
    codex = FakeDecisionModel("codex", make_decision("codex", [proposal]))
    return RunDecisionCycle(
        snapshot_builder=SnapshotBuilder(gateway, ["005930"], clock=lambda: FIXED_NOW),
        consensus_service=ConsensusDecisionService(claude, codex),
        risk_policy=RiskPolicy(
            max_position_weight=Decimal("0.2"),
            max_order_amount=None,
            max_order_ratio=Decimal("0.1"),
            stop_loss_percent=Decimal("10"),
            min_confidence=0.7,
            allowed_codes={"005930"},
            max_proposals_per_cycle=5,
        ),
        watchlist=["005930"],
        ledger=SQLiteDecisionLedger(db_path),
        clock=lambda: FIXED_NOW,
    )


def test_rule_proposal_is_recorded_when_models_fail(tmp_path: Path) -> None:
    db_path = tmp_path / "rule-model-failure.sqlite"

    result = build_rule_cycle(db_path, model_failure=True).run()

    assert result.status is CycleStatus.MODEL_FAILED
    assert result.snapshot is not None
    assert result.snapshot.account.total_asset_value == Decimal("10000")
    assert len(result.verdicts) == 1
    assert result.verdicts[0].proposal.source.value == "rule"
    assert result.verdicts[0].proposal.rule_type.value == "stop_loss"
    assert result.verdicts[0].proposal.quantity == 15
    assert result.verdicts[0].proposal.limit_price * result.verdicts[
        0
    ].proposal.quantity > Decimal("1000")
    assert result.verdicts[0].approved
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT source, rule_type, approved FROM risk_verdicts WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()
    assert row == ("rule", "stop_loss", 1)


def test_rule_proposal_precedes_and_blocks_same_code_consensus(tmp_path: Path) -> None:
    result = build_rule_cycle(tmp_path / "rule-duplicate.sqlite").run()

    assert result.status is CycleStatus.COMPLETED
    assert [item.proposal.source.value for item in result.verdicts] == [
        "rule",
        "consensus",
    ]
    assert result.verdicts[0].approved
    assert not result.verdicts[1].approved
    assert any(
        issue.code.value == "duplicate_code" for issue in result.verdicts[1].reasons
    )
