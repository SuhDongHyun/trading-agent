"""가상 계좌 상태와 모의 체결·평가 이력을 SQLite에 저장한다."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_agent.application.paper_trading import (
    PaperDecisionEvidence,
    PaperFillRow,
    PaperHistory,
    PaperPortfolioStore,
    PaperRecord,
    PaperValuationRow,
)
from trading_agent.domain.models import (
    Action,
    ConsensusBasis,
    ProposalSource,
    RuleType,
    TradeProposal,
)
from trading_agent.domain.paper import PaperHolding, PaperPortfolio

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    initial_cash TEXT NOT NULL,
    cash TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_holdings (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    average_price TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL,
    decision_run_id INTEGER,
    error TEXT,
    model_called INTEGER NOT NULL DEFAULT 0,
    reuse_source_run_id INTEGER,
    reuse_reason TEXT
);
CREATE TABLE IF NOT EXISTS paper_valuations (
    paper_run_id INTEGER PRIMARY KEY REFERENCES paper_runs(id),
    valued_at TEXT NOT NULL,
    cash TEXT NOT NULL,
    holdings_value TEXT NOT NULL,
    total_value TEXT NOT NULL,
    kospi TEXT,
    prices_json TEXT NOT NULL,
    stale_codes_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_run_id INTEGER NOT NULL REFERENCES paper_runs(id),
    executed_at TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_quantity INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    limit_price TEXT NOT NULL,
    quote_price TEXT,
    fill_price TEXT,
    commission TEXT NOT NULL,
    tax TEXT NOT NULL,
    realized_pnl TEXT,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    rule_type TEXT,
    consensus_basis TEXT,
    reason TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS paper_decision_evidence (
    paper_run_id INTEGER PRIMARY KEY REFERENCES paper_runs(id),
    decision_run_id INTEGER NOT NULL,
    judged_at TEXT NOT NULL,
    prices_json TEXT NOT NULL,
    atr14_json TEXT NOT NULL,
    news_ids_json TEXT NOT NULL,
    universe_codes_json TEXT NOT NULL,
    proposals_json TEXT NOT NULL
);
"""


def _dec(value: str | None) -> Decimal | None:
    return None if value is None else Decimal(value)


def _str(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


class SQLitePaperPortfolioStore(PaperPortfolioStore):
    """단일 가상 계좌를 SQLite 파일 하나에 보관한다."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript(_SCHEMA)
            self._migrate_paper_runs(connection)

    @staticmethod
    def _migrate_paper_runs(connection: sqlite3.Connection) -> None:
        """기존 paper DB에 선택적 재사용 기록 열을 추가한다."""

        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(paper_runs)")
        }
        if "model_called" not in columns:
            connection.execute(
                "ALTER TABLE paper_runs ADD COLUMN model_called INTEGER NOT NULL "
                "DEFAULT 0"
            )
            connection.execute(
                "UPDATE paper_runs SET model_called = CASE "
                "WHEN kind = 'cycle' AND status IN ('completed', 'model_failed') "
                "THEN 1 ELSE 0 END"
            )
        if "reuse_source_run_id" not in columns:
            connection.execute(
                "ALTER TABLE paper_runs ADD COLUMN reuse_source_run_id INTEGER"
            )
        if "reuse_reason" not in columns:
            connection.execute("ALTER TABLE paper_runs ADD COLUMN reuse_reason TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def load(self) -> PaperPortfolio | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT initial_cash, cash FROM paper_account WHERE id = 1"
            ).fetchone()
            if row is None:
                return None
            holdings = {
                code: PaperHolding(code, name, quantity, Decimal(average))
                for code, name, quantity, average in connection.execute(
                    "SELECT code, name, quantity, average_price FROM paper_holdings "
                    "ORDER BY rowid"
                )
            }
        return PaperPortfolio(Decimal(row[0]), Decimal(row[1]), holdings)

    def create(self, portfolio: PaperPortfolio, at: datetime) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO paper_account (id, initial_cash, cash, started_at, "
                "updated_at) VALUES (1, ?, ?, ?, ?)",
                (
                    str(portfolio.initial_cash),
                    str(portfolio.cash),
                    at.isoformat(),
                    at.isoformat(),
                ),
            )
            self._write_holdings(connection, portfolio)

    def save(self, portfolio: PaperPortfolio, record: PaperRecord) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE paper_account SET cash = ?, updated_at = ? WHERE id = 1",
                (str(portfolio.cash), record.finished_at.isoformat()),
            )
            self._write_holdings(connection, portfolio)
            cursor = connection.execute(
                "INSERT INTO paper_runs (kind, started_at, finished_at, status, "
                "decision_run_id, error, model_called, reuse_source_run_id, "
                "reuse_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.kind,
                    record.started_at.isoformat(),
                    record.finished_at.isoformat(),
                    record.status,
                    record.run_id,
                    record.error,
                    int(record.model_called),
                    record.reuse_source_run_id,
                    record.reuse_reason,
                ),
            )
            paper_run_id = cursor.lastrowid
            evidence = record.decision_evidence
            if evidence is not None:
                connection.execute(
                    """INSERT OR REPLACE INTO paper_decision_evidence
                       (decision_run_id, paper_run_id, judged_at, prices_json,
                        atr14_json, news_ids_json, universe_codes_json, proposals_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        evidence.run_id,
                        paper_run_id,
                        evidence.judged_at.isoformat(),
                        json.dumps(
                            {
                                code: _str(price)
                                for code, price in evidence.prices.items()
                            }
                        ),
                        json.dumps(
                            {
                                code: _str(value)
                                for code, value in evidence.atr14_percent.items()
                            }
                        ),
                        json.dumps(sorted(evidence.news_ids), ensure_ascii=False),
                        json.dumps(sorted(evidence.universe_codes)),
                        json.dumps(
                            [
                                _proposal_to_json(proposal)
                                for proposal in evidence.proposals
                            ],
                            ensure_ascii=False,
                        ),
                    ),
                )
            valuation = record.valuation
            if valuation is not None:
                connection.execute(
                    "INSERT INTO paper_valuations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        paper_run_id,
                        valuation.at.isoformat(),
                        str(valuation.cash),
                        str(valuation.holdings_value),
                        str(valuation.total_value),
                        _str(valuation.kospi),
                        json.dumps({k: str(v) for k, v in valuation.prices.items()}),
                        json.dumps(list(valuation.stale_codes)),
                    ),
                )
            for fill in record.fills:
                proposal = fill.proposal
                connection.execute(
                    "INSERT INTO paper_fills (paper_run_id, executed_at, code, name, "
                    "action, status, requested_quantity, quantity, limit_price, "
                    "quote_price, fill_price, commission, tax, realized_pnl, "
                    "confidence, source, rule_type, consensus_basis, reason, note) "
                    f"VALUES ({', '.join('?' * 20)})",
                    (
                        paper_run_id,
                        record.finished_at.isoformat(),
                        proposal.code,
                        fill.name,
                        str(proposal.action),
                        str(fill.status),
                        proposal.quantity,
                        fill.quantity,
                        str(proposal.limit_price),
                        _str(fill.quote_price),
                        _str(fill.fill_price),
                        str(fill.commission),
                        str(fill.tax),
                        _str(fill.realized_pnl),
                        proposal.confidence,
                        str(proposal.source),
                        _str_enum(proposal.rule_type),
                        _str_enum(proposal.consensus_basis),
                        proposal.reason,
                        fill.note,
                    ),
                )

    def history(self) -> PaperHistory:
        with closing(self._connect()) as connection:
            account = connection.execute(
                "SELECT started_at FROM paper_account WHERE id = 1"
            ).fetchone()
            if account is None:
                raise LookupError("가상 계좌가 아직 없습니다.")
            valuations = [
                PaperValuationRow(
                    at=datetime.fromisoformat(valued_at),
                    kind=kind,
                    status=status,
                    cash=Decimal(cash),
                    holdings_value=Decimal(holdings_value),
                    total_value=Decimal(total_value),
                    kospi=_dec(kospi),
                    prices={
                        code: Decimal(price)
                        for code, price in json.loads(prices_json).items()
                    },
                )
                for (
                    valued_at,
                    kind,
                    status,
                    cash,
                    holdings_value,
                    total_value,
                    kospi,
                    prices_json,
                ) in connection.execute(
                    "SELECT v.valued_at, r.kind, r.status, v.cash, v.holdings_value, "
                    "v.total_value, v.kospi, v.prices_json FROM paper_valuations v "
                    "JOIN paper_runs r ON r.id = v.paper_run_id ORDER BY r.id"
                )
            ]
            fills = [
                PaperFillRow(
                    at=datetime.fromisoformat(row[0]),
                    code=row[1],
                    name=row[2],
                    action=row[3],
                    status=row[4],
                    requested_quantity=row[5],
                    quantity=row[6],
                    limit_price=Decimal(row[7]),
                    fill_price=_dec(row[8]),
                    commission=Decimal(row[9]),
                    tax=Decimal(row[10]),
                    realized_pnl=_dec(row[11]),
                    source=row[12],
                    note=row[13],
                )
                for row in connection.execute(
                    "SELECT executed_at, code, name, action, status, "
                    "requested_quantity, quantity, limit_price, fill_price, "
                    "commission, tax, realized_pnl, source, note FROM paper_fills "
                    "ORDER BY id"
                )
            ]
            status_counts = dict(
                connection.execute(
                    "SELECT status, COUNT(*) FROM paper_runs WHERE kind = 'cycle' "
                    "GROUP BY status"
                ).fetchall()
            )
            model_call_count = connection.execute(
                "SELECT COUNT(*) FROM paper_runs "
                "WHERE kind = 'cycle' AND model_called = 1"
            ).fetchone()[0]
            reused_count = connection.execute(
                "SELECT COUNT(*) FROM paper_runs "
                "WHERE kind = 'cycle' AND status = 'reused'"
            ).fetchone()[0]
        return PaperHistory(
            started_at=datetime.fromisoformat(account[0]),
            valuations=valuations,
            fills=fills,
            status_counts=status_counts,
            model_call_count=model_call_count,
            reused_count=reused_count,
        )

    def latest_decision_evidence(self) -> PaperDecisionEvidence | None:
        """가장 최근 완료 판단의 비교 데이터와 합의 제안을 읽는다."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT paper_run_id, decision_run_id, judged_at, prices_json,
                          atr14_json, news_ids_json, universe_codes_json,
                          proposals_json
                   FROM paper_decision_evidence ORDER BY paper_run_id DESC LIMIT 1"""
            ).fetchone()
        if row is None:
            return None
        (
            paper_run_id,
            run_id,
            judged_at,
            prices_json,
            atr_json,
            news_json,
            codes_json,
            proposals_json,
        ) = row
        return PaperDecisionEvidence(
            paper_run_id=paper_run_id,
            run_id=run_id,
            judged_at=datetime.fromisoformat(judged_at),
            prices={
                code: _dec(value) for code, value in json.loads(prices_json).items()
            },
            atr14_percent={
                code: _dec(value) for code, value in json.loads(atr_json).items()
            },
            news_ids=frozenset(json.loads(news_json)),
            universe_codes=frozenset(json.loads(codes_json)),
            proposals=tuple(
                _proposal_from_json(item) for item in json.loads(proposals_json)
            ),
        )

    def latest_model_attempt(self) -> tuple[str, int] | None:
        """재사용·일봉 skip을 제외한 마지막 판단 시도를 반환한다."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT status, decision_run_id FROM paper_runs
                   WHERE kind = 'cycle' AND status NOT IN ('reused', 'skipped')
                   ORDER BY id DESC LIMIT 1"""
            ).fetchone()
        if row is None:
            return None
        return row[0], -1 if row[1] is None else row[1]

    def has_filled_since(self, paper_run_id: int) -> bool:
        """모델 판단 실행에서 발생했거나 그 뒤에 발생한 체결이 있는지 본다."""

        with closing(self._connect()) as connection:
            count = connection.execute(
                """SELECT COUNT(*) FROM paper_fills
                   WHERE paper_run_id >= ? AND status = 'filled'""",
                (paper_run_id,),
            ).fetchone()[0]
        return count > 0

    @staticmethod
    def _write_holdings(
        connection: sqlite3.Connection, portfolio: PaperPortfolio
    ) -> None:
        connection.execute("DELETE FROM paper_holdings")
        connection.executemany(
            "INSERT INTO paper_holdings (code, name, quantity, average_price) "
            "VALUES (?, ?, ?, ?)",
            [
                (h.code, h.name, h.quantity, str(h.average_price))
                for h in portfolio.holdings.values()
            ],
        )


def _str_enum(value: object) -> str | None:
    return None if value is None else str(value)


def _proposal_to_json(proposal: TradeProposal) -> dict[str, object]:
    """재사용 판단의 합의 제안을 JSON 원시값으로 바꾼다."""

    return {
        "code": proposal.code,
        "action": proposal.action.value,
        "quantity": proposal.quantity,
        "limit_price": str(proposal.limit_price),
        "confidence": proposal.confidence,
        "reason": proposal.reason,
        "source": proposal.source.value,
        "rule_type": None if proposal.rule_type is None else proposal.rule_type.value,
        "consensus_basis": (
            None if proposal.consensus_basis is None else proposal.consensus_basis.value
        ),
    }


def _proposal_from_json(value: dict[str, object]) -> TradeProposal:
    """저장한 JSON에서 검증된 합의 제안을 복원한다."""

    rule_type = value["rule_type"]
    consensus_basis = value["consensus_basis"]
    return TradeProposal(
        code=str(value["code"]),
        action=Action(str(value["action"])),
        quantity=int(str(value["quantity"])),
        limit_price=Decimal(str(value["limit_price"])),
        confidence=float(str(value["confidence"])),
        reason=str(value["reason"]),
        source=ProposalSource(str(value["source"])),
        rule_type=None if rule_type is None else RuleType(str(rule_type)),
        consensus_basis=(
            None if consensus_basis is None else ConsensusBasis(str(consensus_basis))
        ),
    )
