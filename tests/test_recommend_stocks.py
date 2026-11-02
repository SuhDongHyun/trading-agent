import sqlite3
from datetime import UTC, datetime
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
from trading_agent.domain.models import Action
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
        stop_loss_price=Decimal("90") if priced else None,
        reason="why",
    )


def build(
    db_path: Path,
    gateway: FakeGateway | None = None,
    *,
    codex_fail: bool = False,
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
            rec("005930", Action.BUY, 0.7),
            rec("000660", Action.BUY, 0.8),
            rec("086520", Action.HOLD),
            rec("999999", Action.BUY),
        ],
        fail=codex_fail,
    )
    use_case = RecommendStocks(
        snapshot_builder=SnapshotBuilder(gateway, [], clock=lambda: FIXED_NOW),
        consensus_service=RecommendationConsensusService(claude, codex),
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
    assert [(item.code, item.confidence) for item in result.recommendations] == [
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
    assert (kind, status) == ("recommendation", "completed")
    assert ranked == [(1, "000660", "buy", "120"), (2, "005930", "buy", "120")]
    assert model_items == 8


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


def test_unreadable_quote_is_skipped_but_all_failing_is_data_failure(
    tmp_path: Path,
) -> None:
    partial_gateway = FakeGateway(failing_quotes=frozenset({"086520"}))
    use_case, _, _ = build(tmp_path / "a.sqlite", partial_gateway)

    partial = use_case.run()

    assert partial.status is CycleStatus.COMPLETED
    assert partial.snapshot is not None
    assert [stock.code for stock in partial.snapshot.stocks] == ["005930", "000660"]
    assert "086520:quote" in partial.snapshot.missing

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
        connection.execute("PRAGMA user_version = 1")

    SQLiteDecisionLedger(db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert connection.execute("SELECT kind FROM runs").fetchone() == ("decision",)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"model_recommendations", "consensus_recommendations"} <= tables
