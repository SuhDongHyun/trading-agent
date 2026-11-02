"""SQLite에 사이클과 모델·합의·리스크 결과를 저장한다."""

import json
import sqlite3
from contextlib import closing
from dataclasses import fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from trading_agent.application.decision_ledger import DecisionLedger
from trading_agent.application.ports import ModelRun
from trading_agent.application.recommendation_ledger import RecommendationLedger
from trading_agent.application.records import (
    CycleStatus,
    DecisionCycleResult,
    RecommendationRunResult,
)
from trading_agent.domain.consensus import Disagreement
from trading_agent.domain.models import ModelDecision
from trading_agent.domain.recommendation import (
    ModelRecommendation,
    StockRecommendation,
)

_SCHEMA_VERSION = 2
_RUN_KIND_DECISION = "decision"
_RUN_KIND_RECOMMENDATION = "recommendation"
_MIGRATIONS = {
    1: ("ALTER TABLE runs ADD COLUMN kind TEXT NOT NULL DEFAULT 'decision'",),
}
_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL DEFAULT 'decision',
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        error TEXT,
        prompt TEXT,
        snapshot_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS model_runs (
        run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        model_id TEXT NOT NULL,
        exit_code INTEGER,
        duration_seconds TEXT NOT NULL,
        error TEXT,
        stdout TEXT NOT NULL,
        stderr TEXT NOT NULL,
        response_text TEXT,
        decision_json TEXT,
        PRIMARY KEY (run_id, model_id)
    )""",
    """CREATE TABLE IF NOT EXISTS model_proposals (
        run_id INTEGER NOT NULL,
        model_id TEXT NOT NULL,
        proposal_index INTEGER NOT NULL,
        code TEXT NOT NULL,
        action TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        limit_price TEXT NOT NULL,
        confidence TEXT NOT NULL,
        reason TEXT NOT NULL,
        PRIMARY KEY (run_id, model_id, proposal_index),
        FOREIGN KEY (run_id, model_id)
            REFERENCES model_runs(run_id, model_id) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS consensus_results (
        run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
        summary TEXT NOT NULL,
        claude_summary TEXT NOT NULL,
        codex_summary TEXT NOT NULL,
        missing_info_json TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS consensus_proposals (
        run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        proposal_index INTEGER NOT NULL,
        code TEXT NOT NULL,
        action TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        limit_price TEXT NOT NULL,
        confidence TEXT NOT NULL,
        reason TEXT NOT NULL,
        PRIMARY KEY (run_id, proposal_index)
    )""",
    """CREATE TABLE IF NOT EXISTS consensus_disagreements (
        run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        disagreement_index INTEGER NOT NULL,
        code TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        claude_actions_json TEXT NOT NULL,
        codex_actions_json TEXT NOT NULL,
        PRIMARY KEY (run_id, disagreement_index)
    )""",
    """CREATE TABLE IF NOT EXISTS risk_verdicts (
        run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
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
    )""",
    """CREATE TABLE IF NOT EXISTS model_recommendations (
        run_id INTEGER NOT NULL,
        model_id TEXT NOT NULL,
        item_index INTEGER NOT NULL,
        code TEXT NOT NULL,
        rating TEXT NOT NULL,
        confidence TEXT NOT NULL,
        target_price TEXT,
        stop_loss_price TEXT,
        reason TEXT NOT NULL,
        PRIMARY KEY (run_id, model_id, item_index),
        FOREIGN KEY (run_id, model_id)
            REFERENCES model_runs(run_id, model_id) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS consensus_recommendations (
        run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        rank INTEGER NOT NULL,
        code TEXT NOT NULL,
        rating TEXT NOT NULL,
        confidence TEXT NOT NULL,
        target_price TEXT,
        stop_loss_price TEXT,
        reason TEXT NOT NULL,
        PRIMARY KEY (run_id, rank)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_runs_recent ON runs(started_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(kind, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_risk_approved ON risk_verdicts(approved, run_id)",
    "CREATE INDEX IF NOT EXISTS idx_disagreement_code ON consensus_disagreements(code)",
)


class SQLiteDecisionLedger(DecisionLedger, RecommendationLedger):
    """디렉터리와 버전 스키마를 준비해 매매 사이클과 추천 실행을 기록한다."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def start_cycle(self, started_at: datetime) -> int:
        """사이클을 running으로 저장하고 autoincrement 식별자를 반환한다."""

        return self._start_run(_RUN_KIND_DECISION, started_at)

    def complete_cycle(
        self, result: DecisionCycleResult, finished_at: datetime
    ) -> None:
        """정상 사이클을 한 트랜잭션으로 종료하고 상세 데이터를 저장한다."""

        if result.status is not CycleStatus.COMPLETED:
            raise ValueError("완료 기록에는 completed 결과가 필요합니다.")
        self._finish_cycle(result, finished_at)

    def fail_cycle(self, result: DecisionCycleResult, finished_at: datetime) -> None:
        """실패 사이클을 한 트랜잭션으로 종료하고 확보한 데이터를 저장한다."""

        if result.status is CycleStatus.COMPLETED:
            raise ValueError("실패 기록에는 실패 상태 결과가 필요합니다.")
        self._finish_cycle(result, finished_at)

    def start_recommendation(self, started_at: datetime) -> int:
        """추천 실행을 running으로 저장하고 식별자를 반환한다."""

        return self._start_run(_RUN_KIND_RECOMMENDATION, started_at)

    def finish_recommendation(
        self, result: RecommendationRunResult, finished_at: datetime
    ) -> None:
        """추천 실행과 모델 원문, 합의 의견을 한 트랜잭션으로 저장한다."""

        with self._connection() as connection, connection:
            self._update_run(
                connection,
                result.run_id,
                finished_at,
                result.status,
                result.error,
                result.prompt,
                result.snapshot,
            )
            outcome = result.outcome
            if outcome is None:
                return
            for model_run in (outcome.claude_run, outcome.codex_run):
                self._store_model_run(connection, result.run_id, model_run)
                if model_run.decision is not None:
                    self._store_model_recommendations(
                        connection, result.run_id, model_run.decision
                    )
            consensus = outcome.consensus
            if consensus is None:
                return
            self._store_consensus_summary(
                connection,
                result.run_id,
                consensus.claude_summary,
                consensus.codex_summary,
                consensus.missing_info,
            )
            self._store_disagreements(
                connection, result.run_id, consensus.disagreements
            )
            for rank, item in enumerate(result.recommendations or [], start=1):
                connection.execute(
                    """INSERT INTO consensus_recommendations
                       (run_id, rank, code, rating, confidence, target_price,
                        stop_loss_price, reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (result.run_id, rank, *_recommendation_values(item)),
                )

    def _start_run(self, kind: str, started_at: datetime) -> int:
        """종류별 run을 running으로 저장한다."""

        with self._connection() as connection, connection:
            cursor = connection.execute(
                "INSERT INTO runs(kind, started_at, status) VALUES (?, ?, ?)",
                (kind, _iso_datetime(started_at), "running"),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    @staticmethod
    def _update_run(
        connection: sqlite3.Connection,
        run_id: int,
        finished_at: datetime,
        status: CycleStatus,
        error: str | None,
        prompt: str | None,
        snapshot: object,
    ) -> None:
        """running 상태인 run만 종료 상태로 바꾼다."""

        cursor = connection.execute(
            """UPDATE runs
               SET finished_at = ?, status = ?, error = ?, prompt = ?,
                   snapshot_json = ?
               WHERE id = ? AND status = 'running'""",
            (
                _iso_datetime(finished_at),
                status.value,
                error,
                prompt,
                _json(snapshot),
                run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"running 사이클을 찾을 수 없습니다: {run_id}")

    def _create_schema(self) -> None:
        """최초 실행 시 테이블을 만들고 지원하지 않는 DB 버전을 거절한다."""

        with self._connection() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SCHEMA_VERSION:
                raise RuntimeError(f"SQLite 기록 버전 {version}은 지원하지 않습니다.")
            try:
                connection.execute("BEGIN")
                if version > 0:
                    for from_version in range(version, _SCHEMA_VERSION):
                        for statement in _MIGRATIONS.get(from_version, ()):
                            connection.execute(statement)
                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _finish_cycle(self, result: DecisionCycleResult, finished_at: datetime) -> None:
        """run 행과 관련 상세 행을 같은 SQLite 트랜잭션으로 갱신한다."""

        with self._connection() as connection, connection:
            self._update_run(
                connection,
                result.run_id,
                finished_at,
                result.status,
                result.error,
                result.prompt,
                result.snapshot,
            )

            if result.decision is not None:
                for model_run in (
                    result.decision.claude_run,
                    result.decision.codex_run,
                ):
                    self._store_model_run(connection, result.run_id, model_run)
                    if model_run.decision is not None:
                        self._store_model_proposals(
                            connection,
                            result.run_id,
                            model_run.model_id,
                            model_run.decision,
                        )
                if result.decision.consensus is not None:
                    self._store_consensus(connection, result.run_id, result)
            self._store_risk_verdicts(connection, result.run_id, result)

    @staticmethod
    def _store_model_run(
        connection: sqlite3.Connection, run_id: int, model_run: ModelRun[Any]
    ) -> None:
        """모델 원문과 파싱한 판단을 저장한다."""

        connection.execute(
            """INSERT INTO model_runs
               (run_id, model_id, exit_code, duration_seconds, error, stdout,
                stderr, response_text, decision_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                model_run.model_id,
                model_run.exit_code,
                str(model_run.duration_seconds),
                model_run.error,
                model_run.stdout,
                model_run.stderr,
                model_run.response_text,
                _json(model_run.decision),
            ),
        )

    @staticmethod
    def _store_model_proposals(
        connection: sqlite3.Connection,
        run_id: int,
        model_id: str,
        decision: ModelDecision,
    ) -> None:
        """한 모델의 매매 제안을 저장한다."""

        for index, proposal in enumerate(decision.proposals):
            connection.execute(
                """INSERT INTO model_proposals
                   (run_id, model_id, proposal_index, code, action, quantity,
                    limit_price, confidence, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    model_id,
                    index,
                    proposal.code,
                    proposal.action.value,
                    proposal.quantity,
                    str(proposal.limit_price),
                    str(proposal.confidence),
                    proposal.reason,
                ),
            )

    @staticmethod
    def _store_model_recommendations(
        connection: sqlite3.Connection,
        run_id: int,
        recommendation: ModelRecommendation,
    ) -> None:
        """한 모델의 종목별 투자의견을 저장한다."""

        for index, item in enumerate(recommendation.recommendations):
            connection.execute(
                """INSERT INTO model_recommendations
                   (run_id, model_id, item_index, code, rating, confidence,
                    target_price, stop_loss_price, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    recommendation.model_id,
                    index,
                    *_recommendation_values(item),
                ),
            )

    @staticmethod
    def _store_consensus_summary(
        connection: sqlite3.Connection,
        run_id: int,
        claude_summary: str,
        codex_summary: str,
        missing_info: list[str],
    ) -> None:
        """두 모델 요약과 missing_info 합집합을 저장한다."""

        connection.execute(
            """INSERT INTO consensus_results
               (run_id, summary, claude_summary, codex_summary, missing_info_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                f"claude: {claude_summary}\ncodex: {codex_summary}",
                claude_summary,
                codex_summary,
                _json(missing_info),
            ),
        )

    @staticmethod
    def _store_disagreements(
        connection: sqlite3.Connection,
        run_id: int,
        disagreements: list[Disagreement],
    ) -> None:
        """합의하지 못한 종목과 모델별 의견을 저장한다."""

        for index, disagreement in enumerate(disagreements):
            connection.execute(
                """INSERT INTO consensus_disagreements
                   (run_id, disagreement_index, code, reason_code,
                    claude_actions_json, codex_actions_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    index,
                    disagreement.code,
                    disagreement.reason_code.value,
                    _json(disagreement.claude_actions),
                    _json(disagreement.codex_actions),
                ),
            )

    @staticmethod
    def _store_consensus(
        connection: sqlite3.Connection,
        run_id: int,
        result: DecisionCycleResult,
    ) -> None:
        """합의 제안, 불일치, missing_info 합집합과 요약을 저장한다."""

        assert result.decision is not None
        consensus = result.decision.consensus
        assert consensus is not None
        SQLiteDecisionLedger._store_consensus_summary(
            connection,
            run_id,
            consensus.claude_summary,
            consensus.codex_summary,
            consensus.missing_info,
        )
        for index, proposal in enumerate(consensus.proposals):
            connection.execute(
                """INSERT INTO consensus_proposals
                   (run_id, proposal_index, code, action, quantity, limit_price,
                    confidence, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    index,
                    proposal.code,
                    proposal.action.value,
                    proposal.quantity,
                    str(proposal.limit_price),
                    str(proposal.confidence),
                    proposal.reason,
                ),
            )
        SQLiteDecisionLedger._store_disagreements(
            connection, run_id, consensus.disagreements
        )

    @staticmethod
    def _store_risk_verdicts(
        connection: sqlite3.Connection,
        run_id: int,
        result: DecisionCycleResult,
    ) -> None:
        """리스크 판정과 사유 코드·메시지를 저장한다."""

        for index, verdict in enumerate(result.verdicts or []):
            proposal = verdict.proposal
            connection.execute(
                """INSERT INTO risk_verdicts
                   (run_id, proposal_index, code, action, quantity, limit_price,
                    confidence, reason, approved, skipped, reason_codes_json,
                    reason_messages_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    index,
                    proposal.code,
                    proposal.action.value,
                    proposal.quantity,
                    str(proposal.limit_price),
                    str(proposal.confidence),
                    proposal.reason,
                    int(verdict.approved),
                    int(verdict.skipped),
                    _json([reason.code.value for reason in verdict.reasons]),
                    _json([reason.message for reason in verdict.reasons]),
                ),
            )

    def _connection(self) -> closing[sqlite3.Connection]:
        """외래 키를 적용한 새 연결을 반환한다."""

        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return closing(connection)


def _recommendation_values(item: StockRecommendation) -> tuple[object, ...]:
    """추천 항목을 저장 열 순서(code부터 reason까지)로 바꾼다."""

    return (
        item.code,
        item.rating.value,
        str(item.confidence),
        None if item.target_price is None else str(item.target_price),
        None if item.stop_loss_price is None else str(item.stop_loss_price),
        item.reason,
    )


def _iso_datetime(value: datetime) -> str:
    """저장 시각에 timezone 정보가 포함됐는지 확인한다."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("기록 시각은 timezone-aware datetime이어야 합니다.")
    return value.isoformat()


def _json(value: Any) -> str | None:
    """dataclass, Decimal, 시각, enum을 손실 없이 JSON으로 직렬화한다."""

    if value is None:
        return None
    return json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False)


def _json_value(value: Any) -> Any:
    """JSON으로 표현할 수 없는 도메인 값을 문자열이나 mapping으로 바꾼다."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return _iso_datetime(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [_json_value(item) for item in items]
    return value
