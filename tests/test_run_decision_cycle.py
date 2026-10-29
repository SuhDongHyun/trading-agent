from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_agent.application.consensus_service import ConsensusDecisionService
from trading_agent.application.errors import DecisionModelError, MarketDataUnavailable
from trading_agent.application.ports import (
    DailyBar,
    DecisionModel,
    IndexPoint,
    IndicatorPoint,
    MarketDataGateway,
    MarketIndex,
    ModelRun,
    NewsItem,
    StockQuote,
    TreasuryMaturity,
)
from trading_agent.application.records import CycleStatus
from trading_agent.application.run_decision_cycle import RunDecisionCycle
from trading_agent.application.snapshot_builder import SnapshotBuilder
from trading_agent.domain.models import (
    AccountSnapshot,
    ModelDecision,
    Position,
    TradeProposal,
)
from trading_agent.domain.risk import RiskPolicy
from trading_agent.infrastructure.sqlite_ledger import SQLiteDecisionLedger

FIXED_NOW = datetime(2026, 10, 29, 9, 0, tzinfo=UTC)


class FakeGateway(MarketDataGateway):
    """사이클 테스트에서 사용하는 인메모리 시장 데이터."""

    def __init__(self, fail_account: bool = False) -> None:
        self.fail_account = fail_account

    def get_account(self) -> AccountSnapshot:
        if self.fail_account:
            raise MarketDataUnavailable("account unavailable")
        return AccountSnapshot(Decimal("10000"), [Position("000660", "SK하이닉스", 1)])

    def get_stock_quote(self, code: str) -> StockQuote:
        return StockQuote(code, "테스트 종목", Decimal("100"), Decimal("0"))

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        return []

    def get_moving_average(
        self, code: str, start_date: str, end_date: str, window: int
    ) -> list[IndicatorPoint]:
        return []

    def get_rsi(
        self, code: str, start_date: str, end_date: str, window: int = 14
    ) -> list[IndicatorPoint]:
        return []

    def get_rsi_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        rsi_window: int = 14,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        return []

    def get_macd(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
    ) -> list[IndicatorPoint]:
        return []

    def get_macd_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        return []

    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
        return []

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        return []

    def get_fear_and_greed_index(self) -> IndexPoint:
        return IndexPoint("20261029", Decimal("50"))

    def get_treasury_yield(
        self, maturity: TreasuryMaturity, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        return []


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

    def decide(self, prompt: str) -> ModelRun:
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


def make_proposal(code: str, price: str = "100.123456789") -> TradeProposal:
    return TradeProposal(
        code=code,
        action="buy",
        quantity=2,
        limit_price=Decimal(price),
        confidence=0.9,
        reason="test signal",
    )


def build_cycle(
    db_path: Path,
    *,
    fail_account: bool = False,
    model_failure: bool = False,
) -> RunDecisionCycle:
    gateway = FakeGateway(fail_account)
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

    def clock():
        return FIXED_NOW

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
    result = build_cycle(db_path, **kwargs).run()

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
    assert disagreement == ("000660", "one_sided")
