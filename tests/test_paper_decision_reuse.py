import sqlite3
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fakes import FakeGateway

from trading_agent.application.consensus_service import ConsensusDecisionService
from trading_agent.application.paper_trading import (
    PaperDecisionEvidence,
    PaperRecord,
    RunPaperCycle,
)
from trading_agent.application.ports import ModelRun, StockQuote
from trading_agent.application.records import CycleStatus
from trading_agent.application.run_decision_cycle import RunDecisionCycle
from trading_agent.domain.models import (
    AccountSnapshot,
    Action,
    MarketContext,
    MarketSnapshot,
    ModelDecision,
    NewsSnapshotItem,
    StockSnapshot,
    TradeProposal,
)
from trading_agent.domain.paper import FeeSchedule, PaperPortfolio
from trading_agent.domain.risk import RiskPolicy
from trading_agent.infrastructure.sqlite_ledger import SQLiteDecisionLedger
from trading_agent.infrastructure.sqlite_paper_store import SQLitePaperPortfolioStore

KST = ZoneInfo("Asia/Seoul")
NOW = datetime(2026, 11, 9, 10, 5, tzinfo=KST)
PROPOSAL = TradeProposal("005930", Action.BUY, 2, Decimal(101), 0.9, "근거")


def _snapshot(
    *,
    created_at: datetime = NOW,
    price: Decimal = Decimal(100),
    atr: Decimal | None = Decimal(2),
    codes: tuple[str, ...] = ("005930",),
    news: tuple[NewsSnapshotItem, ...] = (),
    market_session: str = "closed",
) -> MarketSnapshot:
    return MarketSnapshot(
        created_at=created_at,
        account=AccountSnapshot(Decimal(10_000), []),
        stocks=[
            StockSnapshot(
                code,
                f"종목{code}",
                current_price=price,
                atr14_percent=atr,
                news=list(news) if code == "005930" else [],
            )
            for code in codes
        ],
        market_context=MarketContext(),
        missing=[],
        as_of_trading_date="20261109",
        market_session=market_session,
    )


class _ReuseState:
    def __init__(
        self,
        evidence: PaperDecisionEvidence | None,
        *,
        latest_attempt: tuple[str, int] | None = ("completed", 11),
        filled: bool = False,
    ) -> None:
        self.evidence = evidence
        self.attempt = latest_attempt
        self.filled = filled

    def latest_decision_evidence(self):
        return self.evidence

    def latest_model_attempt(self):
        return self.attempt

    def has_filled_since(self, paper_run_id: int) -> bool:
        return self.filled


def _evidence(
    *,
    judged_at: datetime = NOW,
    prices: dict[str, Decimal | None] | None = None,
    atr: dict[str, Decimal | None] | None = None,
    news_ids: frozenset[str] = frozenset(),
    codes: frozenset[str] = frozenset({"005930"}),
) -> PaperDecisionEvidence:
    return PaperDecisionEvidence(
        paper_run_id=7,
        run_id=11,
        judged_at=judged_at,
        prices=prices or {"005930": Decimal(100)},
        atr14_percent=atr or {"005930": Decimal(2)},
        news_ids=news_ids,
        universe_codes=codes,
        proposals=(PROPOSAL,),
    )


def _reuse_cycle(store, **kwargs) -> RunPaperCycle:
    return RunPaperCycle(
        store=store,
        gateway=FakeGateway(),
        cycle_factory=lambda _gateway: None,
        fees=FeeSchedule(Decimal(0), Decimal(0)),
        watchlist=["005930"],
        initial_cash=Decimal(10_000),
        clock=lambda: NOW,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("store", "snapshot", "kwargs", "expected"),
    [
        (_ReuseState(None), _snapshot(), {}, "오늘 완료된 모델 판단이 없음"),
        (
            _ReuseState(_evidence(judged_at=datetime(2026, 11, 8, 15, 0, tzinfo=KST))),
            _snapshot(),
            {},
            "오늘 완료된 모델 판단이 없음",
        ),
        (
            _ReuseState(_evidence(), latest_attempt=("model_failed", 12)),
            _snapshot(),
            {},
            "최근 판단 시도가 model_failed 상태임",
        ),
        (
            _ReuseState(_evidence(), filled=True),
            _snapshot(),
            {},
            "직전 판단 이후 모의 체결이 발생함",
        ),
        (
            _ReuseState(_evidence(atr={"005930": Decimal(1)})),
            _snapshot(price=Decimal(101)),
            {},
            "가격 변동이 ATR 기준",
        ),
        (
            _ReuseState(_evidence(atr={"005930": None})),
            _snapshot(),
            {},
            "직전 가격·ATR 또는 현재가를 알 수 없음",
        ),
        (
            _ReuseState(_evidence(prices={"005930": None})),
            _snapshot(),
            {},
            "직전 가격·ATR 또는 현재가를 알 수 없음",
        ),
        (
            _ReuseState(_evidence(news_ids=frozenset({"key:old"}))),
            _snapshot(
                news=(NewsSnapshotItem("새 기사", "테스트", NOW, 0, False, key="new"),)
            ),
            {},
            "새 뉴스가 있음",
        ),
        (
            _ReuseState(_evidence()),
            _snapshot(codes=("005930", "000660")),
            {},
            "종목 구성이 바뀜",
        ),
        (
            _ReuseState(_evidence()),
            _snapshot(),
            {"force_no_reuse": True},
            "--no-reuse로 강제 호출",
        ),
    ],
)
def test_each_reuse_trigger_requests_model_call(
    store, snapshot, kwargs: dict[str, object], expected: str
) -> None:
    result = _reuse_cycle(store, **kwargs)._reuse_decision(snapshot)

    assert not result.should_reuse
    assert expected in result.reason


def test_unchanged_snapshot_reuses_prior_consensus() -> None:
    result = _reuse_cycle(_ReuseState(_evidence()))._reuse_decision(_snapshot())

    assert result.should_reuse
    assert result.source_run_id == 11
    assert result.proposals == (PROPOSAL,)


def test_news_without_key_uses_title_as_identity() -> None:
    old_title = NewsSnapshotItem("기존 기사", "테스트", NOW, 0, False)
    new_title = NewsSnapshotItem("새 기사", "테스트", NOW, 0, False)
    cycle = _reuse_cycle(
        _ReuseState(_evidence(news_ids=frozenset({"title:기존 기사"})))
    )

    same_title = cycle._reuse_decision(_snapshot(news=(old_title,)))
    new_title_result = cycle._reuse_decision(_snapshot(news=(new_title,)))

    assert same_title.should_reuse
    assert not new_title_result.should_reuse
    assert new_title_result.reason == "새 뉴스가 있음"


class _CountingModel:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.calls = 0

    def decide(self, _prompt: str) -> ModelRun:
        self.calls += 1
        decision = ModelDecision(self.model_id, [PROPOSAL], "summary", [])
        return ModelRun(
            model_id=self.model_id,
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            decision=decision,
        )


class _StaticSnapshotBuilder:
    def __init__(self, gateway, snapshot: MarketSnapshot) -> None:
        self.gateway = gateway
        self.snapshot = snapshot

    def build(self) -> MarketSnapshot:
        return replace(self.snapshot, account=self.gateway.get_account())


class _PriceGateway(FakeGateway):
    def get_stock_quote(self, code: str) -> StockQuote:
        return StockQuote(code, "삼성전자", Decimal(100), Decimal(0))


def _paper_runtime(
    tmp_path: Path,
    *,
    store: SQLitePaperPortfolioStore,
    snapshot: MarketSnapshot,
    claude: _CountingModel,
    codex: _CountingModel,
) -> RunPaperCycle:
    gateway = _PriceGateway()
    ledger_path = tmp_path / "decisions.db"

    def make_cycle(account_gateway):
        return RunDecisionCycle(
            snapshot_builder=_StaticSnapshotBuilder(account_gateway, snapshot),
            consensus_service=ConsensusDecisionService(claude, codex),
            risk_policy=RiskPolicy(
                max_position_weight=Decimal(1),
                max_order_amount=Decimal(1_000),
                min_confidence=0.6,
                allowed_codes={"005930"},
                max_proposals_per_cycle=5,
                max_order_ratio=Decimal(1),
            ),
            watchlist=["005930"],
            ledger=SQLiteDecisionLedger(ledger_path),
            clock=lambda: NOW,
        )

    return RunPaperCycle(
        store=store,
        gateway=gateway,
        cycle_factory=make_cycle,
        fees=FeeSchedule(Decimal(0), Decimal(0)),
        watchlist=["005930"],
        initial_cash=Decimal(10_000),
        clock=lambda: NOW,
    )


def test_reused_cycle_skips_models_and_records_reuse_without_model_runs(
    tmp_path: Path,
) -> None:
    store = SQLitePaperPortfolioStore(tmp_path / "paper.db")
    claude, codex = _CountingModel("claude"), _CountingModel("codex")
    first = _paper_runtime(
        tmp_path,
        store=store,
        snapshot=_snapshot(),
        claude=claude,
        codex=codex,
    ).run()
    second = _paper_runtime(
        tmp_path,
        store=store,
        snapshot=_snapshot(),
        claude=claude,
        codex=codex,
    ).run()

    assert first.result.status is CycleStatus.COMPLETED
    assert second.result.status is CycleStatus.REUSED
    assert (claude.calls, codex.calls) == (1, 1)
    assert second.result.reuse_source_run_id == first.result.run_id
    assert second.result.verdicts and second.result.verdicts[0].approved
    history = store.history()
    assert history.status_counts == {"completed": 1, "reused": 1}
    assert history.model_call_count == 1
    assert history.reused_count == 1

    import sqlite3

    with sqlite3.connect(tmp_path / "decisions.db") as connection:
        reused_status, source_run_id, reason = connection.execute(
            "SELECT status, reuse_source_run_id, error FROM runs WHERE id = ?",
            (second.result.run_id,),
        ).fetchone()
        model_run_count = connection.execute(
            "SELECT COUNT(*) FROM model_runs WHERE run_id = ?",
            (second.result.run_id,),
        ).fetchone()[0]
    assert reused_status == "reused"
    assert source_run_id == first.result.run_id
    assert "변화 없음" in reason
    assert model_run_count == 0


def test_stale_daily_skip_does_not_count_as_attempt_before_first_model_call(
    tmp_path: Path,
) -> None:
    store = SQLitePaperPortfolioStore(tmp_path / "paper.db")
    claude, codex = _CountingModel("claude"), _CountingModel("codex")
    stale_snapshot = replace(
        _snapshot(
            created_at=datetime(2026, 11, 9, 9, 5, tzinfo=KST),
            market_session="holiday",
        ),
        as_of_trading_date="20261106",
    )

    skipped = _paper_runtime(
        tmp_path,
        store=store,
        snapshot=stale_snapshot,
        claude=claude,
        codex=codex,
    ).run()

    assert skipped.result.status is CycleStatus.SKIPPED
    assert (claude.calls, codex.calls) == (0, 0)
    assert store.latest_model_attempt() is None
    assert store.latest_decision_evidence() is None
    assert store.history().status_counts == {"skipped": 1}

    current_snapshot = _snapshot(market_session="open")
    outcome = _paper_runtime(
        tmp_path,
        store=store,
        snapshot=current_snapshot,
        claude=claude,
        codex=codex,
    ).run()

    assert outcome.result.status is CycleStatus.COMPLETED
    assert (claude.calls, codex.calls) == (1, 1)
    assert store.latest_decision_evidence() is not None


def test_stale_daily_skip_preserves_existing_reuse_evidence_and_attempt(
    tmp_path: Path,
) -> None:
    store = SQLitePaperPortfolioStore(tmp_path / "paper.db")
    portfolio = PaperPortfolio(Decimal(10_000), Decimal(10_000))
    store.create(portfolio, NOW)
    store.save(
        portfolio,
        PaperRecord(
            kind="cycle",
            started_at=NOW,
            finished_at=NOW,
            status="completed",
            run_id=42,
            model_called=True,
            decision_evidence=_evidence(
                judged_at=datetime(2026, 11, 8, 15, 0, tzinfo=KST)
            ),
        ),
    )
    evidence_before = store.latest_decision_evidence()
    claude, codex = _CountingModel("claude"), _CountingModel("codex")
    stale_snapshot = replace(
        _snapshot(
            created_at=datetime(2026, 11, 9, 9, 5, tzinfo=KST),
            market_session="holiday",
        ),
        as_of_trading_date="20261106",
    )

    skipped = _paper_runtime(
        tmp_path,
        store=store,
        snapshot=stale_snapshot,
        claude=claude,
        codex=codex,
    ).run()

    assert skipped.result.status is CycleStatus.SKIPPED
    assert store.latest_decision_evidence() == evidence_before
    assert store.latest_model_attempt() == ("completed", 42)
    assert (claude.calls, codex.calls) == (0, 0)


def test_reuse_provider_exception_falls_back_to_model_call(
    tmp_path: Path, monkeypatch
) -> None:
    store = SQLitePaperPortfolioStore(tmp_path / "paper.db")
    claude, codex = _CountingModel("claude"), _CountingModel("codex")
    cycle = _paper_runtime(
        tmp_path,
        store=store,
        snapshot=_snapshot(),
        claude=claude,
        codex=codex,
    )

    def broken_provider(_snapshot: MarketSnapshot):
        raise ValueError("corrupt reuse evidence")

    monkeypatch.setattr(cycle, "_reuse_decision", broken_provider)
    outcome = cycle.run()

    assert outcome.result.status is CycleStatus.COMPLETED
    assert outcome.result.reuse_reason == "재사용 판정 실패: ValueError"
    assert (claude.calls, codex.calls) == (1, 1)


def test_existing_paper_database_gets_backward_compatible_reuse_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old-paper.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE paper_runs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               kind TEXT NOT NULL,
               started_at TEXT NOT NULL,
               finished_at TEXT NOT NULL,
               status TEXT NOT NULL,
               decision_run_id INTEGER,
               error TEXT)"""
        )
        connection.execute(
            """INSERT INTO paper_runs
               (kind, started_at, finished_at, status, decision_run_id)
               VALUES ('cycle', ?, ?, 'completed', 5)""",
            (NOW.isoformat(), NOW.isoformat()),
        )

    SQLitePaperPortfolioStore(path)

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(paper_runs)")
        }
        status, model_called = connection.execute(
            "SELECT status, model_called FROM paper_runs WHERE id = 1"
        ).fetchone()
    assert {"model_called", "reuse_source_run_id", "reuse_reason"} <= columns
    assert status == "completed"
    assert model_called == 1


def test_version_seven_decision_ledger_migrates_reuse_source_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "version-seven.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE runs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               kind TEXT NOT NULL,
               started_at TEXT NOT NULL,
               finished_at TEXT,
               status TEXT NOT NULL,
               error TEXT,
               prompt TEXT,
               snapshot_json TEXT)"""
        )
        connection.execute(
            "INSERT INTO runs (kind, started_at, status) "
            "VALUES ('decision', ?, 'completed')",
            (NOW.isoformat(),),
        )
        connection.execute("PRAGMA user_version = 7")

    SQLiteDecisionLedger(path)

    with sqlite3.connect(path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        status = connection.execute("SELECT status FROM runs WHERE id = 1").fetchone()[
            0
        ]
    assert version == 8
    assert "reuse_source_run_id" in columns
    assert status == "completed"


def test_reused_consensus_is_rechecked_and_can_fill_then_triggers_model(
    tmp_path: Path,
) -> None:
    store = SQLitePaperPortfolioStore(tmp_path / "paper.db")
    portfolio = PaperPortfolio(Decimal(10_000), Decimal(10_000))
    store.create(portfolio, NOW)
    store.save(
        portfolio,
        PaperRecord(
            kind="cycle",
            started_at=NOW,
            finished_at=NOW,
            status="completed",
            run_id=42,
            model_called=True,
            decision_evidence=_evidence(),
        ),
    )
    claude, codex = _CountingModel("claude"), _CountingModel("codex")
    cycle = _paper_runtime(
        tmp_path,
        store=store,
        snapshot=_snapshot(market_session="open"),
        claude=claude,
        codex=codex,
    )

    reused = cycle.run()

    assert reused.result.status is CycleStatus.REUSED
    assert reused.result.verdicts and reused.result.verdicts[0].approved
    assert reused.fills[0].status.value == "filled"
    assert reused.portfolio.holdings["005930"].quantity == 2
    assert (claude.calls, codex.calls) == (0, 0)
    assert store.has_filled_since(1)

    next_result = _paper_runtime(
        tmp_path,
        store=store,
        snapshot=_snapshot(market_session="closed"),
        claude=claude,
        codex=codex,
    ).run()

    assert next_result.result.status is CycleStatus.COMPLETED
    assert (claude.calls, codex.calls) == (1, 1)
