"""가상 계좌 상태와 모의 체결·평가 이력을 SQLite에 저장한다."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_agent.application.paper_trading import (
    PaperFillRow,
    PaperHistory,
    PaperPortfolioStore,
    PaperRecord,
    PaperValuationRow,
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
    error TEXT
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
                "decision_run_id, error) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.kind,
                    record.started_at.isoformat(),
                    record.finished_at.isoformat(),
                    record.status,
                    record.run_id,
                    record.error,
                ),
            )
            paper_run_id = cursor.lastrowid
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
        return PaperHistory(
            started_at=datetime.fromisoformat(account[0]),
            valuations=valuations,
            fills=fills,
            status_counts=status_counts,
        )

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
