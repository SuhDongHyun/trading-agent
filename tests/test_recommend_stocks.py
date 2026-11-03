import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from fakes import FakeGateway

from trading_agent.application.consensus_service import (
    RecommendationConsensusService,
)
from trading_agent.application.errors import DecisionModelError
from trading_agent.application.ports import DecisionModel, ModelRun
from trading_agent.application.recommend_stocks import RecommendStocks
from trading_agent.application.records import CycleStatus
from trading_agent.application.snapshot_builder import SnapshotBuilder
from trading_agent.domain.models import (
    Action,
    ConfidenceAggregation,
    ConsensusBasis,
    ConsensusPolicyName,
)
from trading_agent.domain.recommendation import (
    ModelRecommendation,
    StockRecommendation,
)
from trading_agent.infrastructure.sqlite_ledger import SQLiteDecisionLedger

FIXED_NOW = datetime(2026, 11, 2, 0, 30, tzinfo=UTC)


class FakeRecommender(DecisionModel[ModelRecommendation]):
    def __init__(
        self, model_id: str, items: list[StockRecommendation], fail: bool = False
    ) -> None:
        self.model_id = model_id
        self.items = items
        self.fail = fail
        self.prompts: list[str] = []

    def decide(self, prompt: str) -> ModelRun[ModelRecommendation]:
        self.prompts.append(prompt)
        if self.fail:
            run: ModelRun[ModelRecommendation] = ModelRun(
                self.model_id, "partial", "boom", 1, 0.2, error="CLI exit code 1"
            )
            raise DecisionModelError("failed", run)
        return ModelRun(
            model_id=self.model_id,
            stdout="raw",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            decision=ModelRecommendation(
                self.model_id, self.items, f"{self.model_id} view", ["flows"]
            ),
        )


def rec(code: str, rating: Action, confidence: float = 0.8) -> StockRecommendation:
    priced = rating is not Action.HOLD
    return StockRecommendation(
        code=code,
        rating=rating,
        confidence=confidence,
        target_price=Decimal("120") if priced else None,
        invalidation_price=Decimal("90") if priced else None,
        reason="why",
    )


def build(
    db_path: Path,
    gateway: FakeGateway | None = None,
    *,
    codex_fail: bool = False,
    policy: ConsensusPolicyName = ConsensusPolicyName.STRICT,
    confidence_aggregation: ConfidenceAggregation = ConfidenceAggregation.MIN,
    codex_005930_rating: Action = Action.BUY,
) -> tuple[RecommendStocks, FakeGateway, FakeRecommender]:
    gateway = gateway or FakeGateway()
    claude = FakeRecommender(
        "claude",
        [
            rec("005930", Action.BUY, 0.6),
            rec("000660", Action.BUY, 0.9),
            rec("086520", Action.SELL),
            rec("999999", Action.BUY),
        ],
    )
    codex = FakeRecommender(
        "codex",
        [
            rec("005930", codex_005930_rating, 0.7),
            rec("000660", Action.BUY, 0.8),
            rec("086520", Action.HOLD),
            rec("999999", Action.HOLD),
        ],
        fail=codex_fail,
    )
    use_case = RecommendStocks(
        snapshot_builder=SnapshotBuilder(gateway, [], clock=lambda: FIXED_NOW),
        consensus_service=RecommendationConsensusService(
            claude,
            codex,
            policy=policy,
            confidence_aggregation=confidence_aggregation,
        ),
        universe=["005930", "000660", "086520"],
        ledger=SQLiteDecisionLedger(db_path),
        clock=lambda: FIXED_NOW,
    )
    return use_case, gateway, claude


def test_completed_run_ranks_agreed_buys_and_ignores_codes_outside_snapshot(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ledger.sqlite"
    use_case, gateway, claude = build(db_path)

    result = use_case.run()

    assert result.status is CycleStatus.COMPLETED
    assert gateway.account_reads == 0
    assert [(item.code, item.confidence) for item in result.recommendations or []] == [
        ("000660", 0.8),
        ("005930", 0.6),
    ]
    assert result.outcome is not None and result.outcome.consensus is not None
    assert [d.code for d in result.outcome.consensus.disagreements] == ["086520"]
    assert "계좌 잔고나 보유 여부와 무관하게" in claude.prompts[0]

    with sqlite3.connect(db_path) as connection:
        kind, status = connection.execute(
            "SELECT kind, status FROM runs WHERE id = ?", (result.run_id,)
        ).fetchone()
        ranked = connection.execute(
            """SELECT rank, code, rating, target_price FROM consensus_recommendations
               WHERE run_id = ? ORDER BY rank""",
            (result.run_id,),
        ).fetchall()
        model_items = connection.execute(
            "SELECT COUNT(*) FROM model_recommendations WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()[0]
        disagreements = connection.execute(
            "SELECT code FROM consensus_disagreements WHERE run_id = ? "
            "ORDER BY disagreement_index",
            (result.run_id,),
        ).fetchall()
    assert (kind, status) == ("recommendation", "completed")
    assert ranked == [(1, "000660", "buy", "120"), (2, "005930", "buy", "120")]
    assert model_items == 8
    assert disagreements == [("086520",)]


def test_recommendation_one_sided_basis_and_policy_are_recorded(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "one-sided.sqlite"
    use_case, _, _ = build(
        db_path,
        policy=ConsensusPolicyName.BUY_ONE_SIDE,
        confidence_aggregation=ConfidenceAggregation.MEAN,
        codex_005930_rating=Action.HOLD,
    )

    result = use_case.run()

    assert result.outcome is not None and result.outcome.consensus is not None
    adopted = next(
        item for item in result.recommendations or [] if item.code == "005930"
    )
    assert adopted.consensus_basis is ConsensusBasis.CLAUDE_ONLY
    assert adopted.confidence == 0.6
    assert adopted.target_price == Decimal("120")
    assert "[codex:hold] why" in adopted.reason
    with sqlite3.connect(db_path) as connection:
        policy, aggregation = connection.execute(
            "SELECT policy, confidence_aggregation FROM consensus_results "
            "WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()
        basis = connection.execute(
            "SELECT consensus_basis FROM consensus_recommendations "
            "WHERE run_id = ? AND code = '005930'",
            (result.run_id,),
        ).fetchone()[0]
    assert (policy, aggregation, basis) == ("buy_one_side", "mean", "claude_only")


def test_model_failure_fails_closed_and_keeps_both_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite"
    use_case, _, _ = build(db_path, codex_fail=True)

    result = use_case.run()

    assert result.status is CycleStatus.MODEL_FAILED
    assert result.recommendations == []
    assert "codex: CLI exit code 1" in (result.error or "")
    with sqlite3.connect(db_path) as connection:
        models = connection.execute(
            "SELECT model_id, exit_code FROM model_runs WHERE run_id = ?",
            (result.run_id,),
        ).fetchall()
    assert sorted(models) == [("claude", 0), ("codex", 1)]


def test_unexpected_error_is_recorded_with_error_status(tmp_path: Path) -> None:
    db_path = tmp_path / "error.sqlite"
    use_case, _, _ = build(db_path)

    def fail_prompt(snapshot) -> str:
        raise RuntimeError("prompt failed")

    use_case.prompt_builder = fail_prompt
    result = use_case.run()

    assert result.status is CycleStatus.ERROR
    assert "RuntimeError: prompt failed" in (result.error or "")
    with sqlite3.connect(db_path) as connection:
        status, error = connection.execute(
            "SELECT status, error FROM runs WHERE id = ?", (result.run_id,)
        ).fetchone()
    assert status == "error"
    assert "RuntimeError: prompt failed" in error


def test_interruption_closes_recommendation_run_then_reraises(tmp_path: Path) -> None:
    db_path = tmp_path / "interrupted.sqlite"
    use_case, _, _ = build(db_path)

    class SimulatedInterrupt(BaseException):
        pass

    def interrupt(snapshot) -> str:
        raise SimulatedInterrupt("stopped")

    use_case.prompt_builder = interrupt
    with pytest.raises(SimulatedInterrupt, match="stopped"):
        use_case.run()

    with sqlite3.connect(db_path) as connection:
        status, error = connection.execute("SELECT status, error FROM runs").fetchone()
    assert status == "error"
    assert "SimulatedInterrupt: stopped" in error


def test_recommendation_recording_failure_falls_back_to_error_run(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "recording-error.sqlite"
    use_case, _, _ = build(db_path)
    ledger = use_case.ledger
    finish = ledger.finish_recommendation

    def fail_first_finish(result, finished_at) -> None:
        if result.status is CycleStatus.COMPLETED:
            raise RuntimeError("recommendation write failed")
        finish(result, finished_at)

    monkeypatch.setattr(ledger, "finish_recommendation", fail_first_finish)
    result = use_case.run()

    assert result.status is CycleStatus.ERROR
    assert "recommendation write failed" in (result.error or "")
    with sqlite3.connect(db_path) as connection:
        status, error = connection.execute(
            "SELECT status, error FROM runs WHERE id = ?", (result.run_id,)
        ).fetchone()
    assert status == "error"
    assert "recommendation write failed" in error


def test_unreadable_quote_is_skipped_but_all_failing_is_data_failure(
    tmp_path: Path,
) -> None:
    partial_gateway = FakeGateway(failing_quotes=frozenset({"086520"}))
    use_case, _, _ = build(tmp_path / "a.sqlite", partial_gateway)

    partial = use_case.run()

    assert partial.status is CycleStatus.COMPLETED
    assert partial.snapshot is not None
    assert [stock.code for stock in partial.snapshot.stocks] == ["005930", "000660"]
    assert "086520:quote(unavailable)" in partial.snapshot.missing

    all_failing = FakeGateway(failing_quotes=frozenset({"005930", "000660", "086520"}))
    use_case, _, _ = build(tmp_path / "b.sqlite", all_failing)

    failed = use_case.run()

    assert failed.status is CycleStatus.DATA_FAILED
    with sqlite3.connect(tmp_path / "b.sqlite") as connection:
        assert connection.execute("SELECT status FROM runs").fetchone() == (
            "data_failed",
        )


def test_empty_universe_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RecommendStocks(
            snapshot_builder=SnapshotBuilder(FakeGateway(), []),
            consensus_service=RecommendationConsensusService(
                FakeRecommender("claude", []), FakeRecommender("codex", [])
            ),
            universe=[],
            ledger=SQLiteDecisionLedger(tmp_path / "x.sqlite"),
            clock=lambda: FIXED_NOW,
        )


def test_version_1_ledger_is_migrated_in_place(tmp_path: Path) -> None:
    db_path = tmp_path / "v1.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                error TEXT,
                prompt TEXT,
                snapshot_json TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO runs(started_at, status) VALUES ('2026-11-01T00:00:00+09:00',"
            " 'completed')"
        )
        connection.execute(
            """CREATE TABLE risk_verdicts (
                run_id INTEGER NOT NULL,
                proposal_index INTEGER NOT NULL,
                code TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                limit_price TEXT NOT NULL,
                confidence TEXT NOT NULL,
                reason TEXT NOT NULL,
                approved INTEGER NOT NULL,
                skipped INTEGER NOT NULL,
                reason_codes_json TEXT NOT NULL,
                reason_messages_json TEXT NOT NULL,
                PRIMARY KEY (run_id, proposal_index)
            )"""
        )
        connection.execute(
            """INSERT INTO risk_verdicts VALUES
               (1, 0, '005930', 'buy', 1, '100', '0.8', '기존', 1, 0, '[]', '[]')"""
        )
        connection.execute("PRAGMA user_version = 1")

    SQLiteDecisionLedger(db_path)
    SQLiteDecisionLedger(db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (5,)
        assert connection.execute(
            "SELECT kind, started_at, status FROM runs"
        ).fetchone() == (
            "decision",
            "2026-11-01T00:00:00+09:00",
            "completed",
        )
        assert connection.execute(
            "SELECT source, rule_type FROM risk_verdicts"
        ).fetchone() == ("consensus", None)
        assert connection.execute(
            "SELECT code, action, quantity, limit_price, reason, approved, skipped "
            "FROM risk_verdicts"
        ).fetchone() == ("005930", "buy", 1, "100", "기존", 1, 0)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"model_recommendations", "consensus_recommendations"} <= tables


def test_version_2_ledger_preserves_verdicts_and_defaults_source_to_consensus(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "v2.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE runs (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'decision',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                error TEXT,
                prompt TEXT,
                snapshot_json TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO runs(id, started_at, status) VALUES "
            "(7, '2026-11-01', 'completed')"
        )
        connection.execute(
            """CREATE TABLE risk_verdicts (
                run_id INTEGER NOT NULL,
                proposal_index INTEGER NOT NULL,
                code TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                limit_price TEXT NOT NULL,
                confidence TEXT NOT NULL,
                reason TEXT NOT NULL,
                approved INTEGER NOT NULL,
                skipped INTEGER NOT NULL,
                reason_codes_json TEXT NOT NULL,
                reason_messages_json TEXT NOT NULL,
                PRIMARY KEY (run_id, proposal_index)
            )"""
        )
        connection.execute(
            """INSERT INTO risk_verdicts VALUES
               (7, 0, '005930', 'sell', 2, '100', '0.9', '기존 판정',
                1, 0, '[]', '[]')"""
        )
        connection.execute("PRAGMA user_version = 2")

    SQLiteDecisionLedger(db_path)
    SQLiteDecisionLedger(db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (5,)
        assert connection.execute(
            "SELECT kind, started_at, status FROM runs WHERE id = 7"
        ).fetchone() == ("decision", "2026-11-01", "completed")
        row = connection.execute(
            """SELECT run_id, proposal_index, code, action, quantity, limit_price,
                      reason, approved, skipped, source, rule_type
               FROM risk_verdicts"""
        ).fetchone()
    assert row == (
        7,
        0,
        "005930",
        "sell",
        2,
        "100",
        "기존 판정",
        1,
        0,
        "consensus",
        None,
    )


def test_version_3_recommendation_prices_are_renamed_and_reopen_is_idempotent(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "v3.sqlite"
    old_column = "stop_loss" + "_price"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE runs (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'decision',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                error TEXT,
                prompt TEXT,
                snapshot_json TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO runs(id, kind, started_at, status) VALUES "
            "(9, 'recommendation', '2026-11-02T00:00:00+00:00', 'completed')"
        )
        connection.execute(
            """CREATE TABLE model_runs (
                run_id INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                exit_code INTEGER,
                duration_seconds TEXT NOT NULL,
                error TEXT,
                stdout TEXT NOT NULL,
                stderr TEXT NOT NULL,
                response_text TEXT,
                decision_json TEXT,
                PRIMARY KEY (run_id, model_id)
            )"""
        )
        connection.execute(
            "INSERT INTO model_runs(run_id, model_id, duration_seconds, stdout, "
            "stderr) "
            "VALUES (9, 'claude', '0.1', '', '')"
        )
        connection.execute(
            f"""CREATE TABLE model_recommendations (
                run_id INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                item_index INTEGER NOT NULL,
                code TEXT NOT NULL,
                rating TEXT NOT NULL,
                confidence TEXT NOT NULL,
                target_price TEXT,
                {old_column} TEXT,
                reason TEXT NOT NULL,
                PRIMARY KEY (run_id, model_id, item_index),
                FOREIGN KEY (run_id, model_id)
                    REFERENCES model_runs(run_id, model_id) ON DELETE CASCADE
            )"""
        )
        connection.execute(
            "INSERT INTO model_recommendations VALUES "
            "(9, 'claude', 0, '005930', 'buy', '0.8', '120', '90', '기존 추천')"
        )
        connection.execute(
            f"""CREATE TABLE consensus_recommendations (
                run_id INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                code TEXT NOT NULL,
                rating TEXT NOT NULL,
                confidence TEXT NOT NULL,
                target_price TEXT,
                {old_column} TEXT,
                reason TEXT NOT NULL,
                PRIMARY KEY (run_id, rank),
                FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
            )"""
        )
        connection.execute(
            "INSERT INTO consensus_recommendations VALUES "
            "(9, 1, '005930', 'buy', '0.8', '120', '90', '기존 합의')"
        )
        connection.execute("PRAGMA user_version = 3")

    SQLiteDecisionLedger(db_path)
    SQLiteDecisionLedger(db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (5,)
        model_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(model_recommendations)")
        }
        consensus_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(consensus_recommendations)"
            )
        }
        model_row = connection.execute(
            "SELECT run_id, model_id, code, target_price, invalidation_price, reason "
            "FROM model_recommendations"
        ).fetchone()
        consensus_row = connection.execute(
            "SELECT run_id, rank, code, target_price, invalidation_price, reason "
            "FROM consensus_recommendations"
        ).fetchone()
    assert "invalidation_price" in model_columns and old_column not in model_columns
    assert (
        "invalidation_price" in consensus_columns
        and old_column not in consensus_columns
    )
    assert model_row == (9, "claude", "005930", "120", "90", "기존 추천")
    assert consensus_row == (9, 1, "005930", "120", "90", "기존 합의")


def test_version_4_migrates_policy_defaults_and_preserves_rows(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "v4.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE runs (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'decision',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                error TEXT,
                prompt TEXT,
                snapshot_json TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO runs(id, started_at, status) VALUES "
            "(11, '2026-11-02T00:00:00+00:00', 'completed')"
        )
        connection.execute(
            """CREATE TABLE consensus_results (
                run_id INTEGER PRIMARY KEY,
                summary TEXT NOT NULL,
                claude_summary TEXT NOT NULL,
                codex_summary TEXT NOT NULL,
                missing_info_json TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO consensus_results VALUES (11, '요약', 'c', 'x', '[]')"
        )
        connection.execute(
            """CREATE TABLE consensus_proposals (
                run_id INTEGER NOT NULL,
                proposal_index INTEGER NOT NULL,
                code TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                limit_price TEXT NOT NULL,
                confidence TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (run_id, proposal_index)
            )"""
        )
        connection.execute(
            "INSERT INTO consensus_proposals VALUES "
            "(11, 0, '005930', 'buy', 2, '100', '0.8', '기존 제안')"
        )
        connection.execute(
            """CREATE TABLE consensus_recommendations (
                run_id INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                code TEXT NOT NULL,
                rating TEXT NOT NULL,
                confidence TEXT NOT NULL,
                target_price TEXT,
                invalidation_price TEXT,
                reason TEXT NOT NULL,
                PRIMARY KEY (run_id, rank)
            )"""
        )
        connection.execute(
            "INSERT INTO consensus_recommendations VALUES "
            "(11, 1, '000660', 'buy', '0.9', '120', '90', '기존 추천')"
        )
        connection.execute("PRAGMA user_version = 4")

    SQLiteDecisionLedger(db_path)
    SQLiteDecisionLedger(db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (5,)
        result = connection.execute(
            "SELECT summary, policy, confidence_aggregation "
            "FROM consensus_results WHERE run_id = 11"
        ).fetchone()
        proposal_row = connection.execute(
            "SELECT code, reason, consensus_basis FROM consensus_proposals "
            "WHERE run_id = 11"
        ).fetchone()
        recommendation_row = connection.execute(
            "SELECT code, reason, consensus_basis FROM consensus_recommendations "
            "WHERE run_id = 11"
        ).fetchone()
    assert result == ("요약", "strict", "min")
    assert proposal_row == ("005930", "기존 제안", "agreed")
    assert recommendation_row == ("000660", "기존 추천", "agreed")


def test_open_marks_only_running_rows_older_than_configured_threshold_abandoned(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "stale.sqlite"
    SQLiteDecisionLedger(db_path)
    now = datetime.now(UTC)
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            "INSERT INTO runs(kind, started_at, status) VALUES (?, ?, 'running')",
            [
                ("decision", (now - timedelta(hours=2)).isoformat()),
                ("recommendation", (now - timedelta(minutes=30)).isoformat()),
            ],
        )

    SQLiteDecisionLedger(db_path, stale_run_after=timedelta(hours=1))

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT kind, status, finished_at, error FROM runs ORDER BY id"
        ).fetchall()
    assert rows[0][0] == "decision"
    assert rows[0][1] == "abandoned"
    assert datetime.fromisoformat(rows[0][2]).tzinfo is not None
    assert "abandoned" in rows[0][3]
    assert rows[1] == ("recommendation", "running", None, None)


def test_stale_run_threshold_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stale_run_after"):
        SQLiteDecisionLedger(tmp_path / "invalid.sqlite", stale_run_after=timedelta(0))
