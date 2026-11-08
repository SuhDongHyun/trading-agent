from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fakes import FakeGateway

from trading_agent.application.paper_trading import (
    MarkPaperPortfolio,
    PaperAccountGateway,
    RunPaperCycle,
    build_paper_report,
)
from trading_agent.application.ports import IndexPoint, MarketIndex, StockQuote
from trading_agent.application.records import CycleStatus, DecisionCycleResult
from trading_agent.domain.models import (
    Action,
    MarketContext,
    MarketSnapshot,
    ProposalSource,
    RuleType,
    StockSnapshot,
    TradeProposal,
)
from trading_agent.domain.paper import (
    FeeSchedule,
    FillStatus,
    PaperHolding,
    PaperPortfolio,
)
from trading_agent.domain.risk import RiskVerdict
from trading_agent.infrastructure.sqlite_paper_store import SQLitePaperPortfolioStore

KST = ZoneInfo("Asia/Seoul")
FEES = FeeSchedule(Decimal("0.00015"), Decimal("0.002"))


def _buy(code: str = "005930", quantity: int = 10, limit: str = "100000"):
    return TradeProposal(code, Action.BUY, quantity, Decimal(limit), 0.8, "근거")


def _sell(code: str = "005930", quantity: int = 10, limit: str = "100000"):
    return TradeProposal(code, Action.SELL, quantity, Decimal(limit), 0.8, "근거")


def test_buy_fills_at_quote_with_commission_and_updates_average() -> None:
    portfolio = PaperPortfolio(Decimal(10_000_000), Decimal(10_000_000))

    fill = portfolio.execute(_buy(quantity=10), "삼성전자", Decimal(99_000), FEES)

    assert fill.status is FillStatus.FILLED
    assert fill.fill_price == Decimal(99_000)
    assert fill.commission == Decimal(148)  # 990,000 x 0.015% = 148.5 -> 148
    assert portfolio.cash == Decimal(10_000_000 - 990_000 - 148)
    second = portfolio.execute(_buy(quantity=10), "삼성전자", Decimal(100_000), FEES)
    assert second.status is FillStatus.FILLED
    holding = portfolio.holdings["005930"]
    assert holding.quantity == 20
    assert holding.average_price == Decimal(99_500)


def test_buy_above_limit_is_unfilled_and_leaves_cash() -> None:
    portfolio = PaperPortfolio(Decimal(1_000_000), Decimal(1_000_000))

    fill = portfolio.execute(_buy(limit="98000"), "삼성전자", Decimal(99_000), FEES)

    assert fill.status is FillStatus.UNFILLED
    assert portfolio.cash == Decimal(1_000_000)
    assert portfolio.holdings == {}


def test_buy_quantity_is_reduced_to_affordable_cash() -> None:
    portfolio = PaperPortfolio(Decimal(1_000_000), Decimal(300_000))

    fill = portfolio.execute(_buy(quantity=10), "삼성전자", Decimal(100_000), FEES)

    assert fill.quantity == 2
    assert portfolio.cash == Decimal(300_000 - 200_000 - 30)


def test_sell_applies_tax_and_realizes_pnl_and_removes_holding() -> None:
    portfolio = PaperPortfolio(
        Decimal(10_000_000),
        Decimal(0),
        {"005930": PaperHolding("005930", "삼성전자", 10, Decimal(90_000))},
    )

    fill = portfolio.execute(
        _sell(quantity=10, limit="99000"), "삼성전자", Decimal(100_000), FEES
    )

    assert fill.status is FillStatus.FILLED
    assert fill.commission == Decimal(150)
    assert fill.tax == Decimal(2_000)
    assert fill.realized_pnl == Decimal(100_000 - 150 - 2_000)
    assert portfolio.cash == Decimal(1_000_000 - 150 - 2_000)
    assert "005930" not in portfolio.holdings


def test_consensus_sell_below_limit_is_unfilled_but_rule_sell_is_market() -> None:
    holding = PaperHolding("005930", "삼성전자", 10, Decimal(90_000))
    portfolio = PaperPortfolio(Decimal(1), Decimal(0), {"005930": holding})

    unfilled = portfolio.execute(
        _sell(limit="101000"), "삼성전자", Decimal(100_000), FEES
    )
    rule = TradeProposal(
        "005930",
        Action.SELL,
        10,
        Decimal(101_000),
        1.0,
        "손절",
        source=ProposalSource.RULE,
        rule_type=RuleType.STOP_LOSS,
    )
    filled = portfolio.execute(rule, "삼성전자", Decimal(100_000), FEES)

    assert unfilled.status is FillStatus.UNFILLED
    assert filled.status is FillStatus.FILLED
    assert filled.fill_price == Decimal(100_000)


def test_paper_gateway_replaces_only_account() -> None:
    portfolio = PaperPortfolio(
        Decimal(5_000_000),
        Decimal(4_000_000),
        {"005930": PaperHolding("005930", "삼성전자", 10, Decimal(100_000))},
    )
    delegate = FakeGateway()
    gateway = PaperAccountGateway(delegate, portfolio)

    account = gateway.get_account()

    assert delegate.account_reads == 0
    assert account.cash == Decimal(4_000_000)
    assert account.positions[0].average_price == Decimal(100_000)
    assert gateway.get_stock_quote("005930").current_price == Decimal(100)


class _QuoteGateway(FakeGateway):
    def __init__(self, prices: dict[str, Decimal], kospi: Decimal) -> None:
        super().__init__()
        self.prices = prices
        self.kospi = kospi

    def get_stock_quote(self, code: str) -> StockQuote:
        return StockQuote(code, f"종목{code}", self.prices[code], Decimal(0))

    def get_market_index(self, index, start_date, end_date):
        assert index is MarketIndex.KOSPI
        return [IndexPoint("20261108", Decimal(1)), IndexPoint(end_date, self.kospi)]


class _FakeCycle:
    def __init__(
        self,
        gateway,
        proposals: list[TradeProposal],
        session: str = "open",
        status: CycleStatus = CycleStatus.COMPLETED,
    ) -> None:
        self.gateway = gateway
        self.proposals = proposals
        self.session = session
        self.status = status

    def run(self) -> DecisionCycleResult:
        account = self.gateway.get_account()
        snapshot = MarketSnapshot(
            created_at=datetime(2026, 11, 9, 10, 5, tzinfo=KST),
            account=account,
            stocks=[StockSnapshot("005930", "삼성전자")],
            market_context=MarketContext(),
            missing=[],
            market_session=self.session,
        )
        return DecisionCycleResult(
            run_id=1,
            status=self.status,
            snapshot=snapshot,
            verdicts=[RiskVerdict(p, True, []) for p in self.proposals]
            + [RiskVerdict(_buy("000660"), False, [])],
        )


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 9, 10, 5, tzinfo=KST)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _paper_cycle(tmp_path: Path, gateway, proposals, **cycle_kwargs) -> RunPaperCycle:
    return RunPaperCycle(
        store=SQLitePaperPortfolioStore(tmp_path / "paper.db"),
        gateway=gateway,
        cycle_factory=lambda g: _FakeCycle(g, proposals, **cycle_kwargs),
        fees=FEES,
        watchlist=["005930", "000660"],
        initial_cash=Decimal(10_000_000),
        clock=_Clock(),
    )


def test_run_paper_cycle_executes_approved_and_persists(tmp_path: Path) -> None:
    gateway = _QuoteGateway(
        {"005930": Decimal(100_000), "000660": Decimal(200_000)}, Decimal(2_500)
    )

    outcome = _paper_cycle(tmp_path, gateway, [_buy(quantity=5)]).run()

    assert [fill.status for fill in outcome.fills] == [FillStatus.FILLED]
    assert outcome.fills[0].name == "삼성전자"
    assert outcome.valuation is not None
    assert outcome.valuation.holdings_value == Decimal(500_000)
    assert outcome.valuation.kospi == Decimal(2_500)
    assert outcome.valuation.prices["000660"] == Decimal(200_000)
    store = SQLitePaperPortfolioStore(tmp_path / "paper.db")
    loaded = store.load()
    assert loaded is not None
    assert loaded.cash == Decimal(10_000_000 - 500_000 - 75)
    assert loaded.holdings["005930"].quantity == 5
    history = store.history()
    assert history.status_counts == {"completed": 1}
    assert len(history.fills) == 1
    assert history.valuations[0].total_value == Decimal(10_000_000 - 75)


def test_run_paper_cycle_does_not_fill_outside_market_hours(tmp_path: Path) -> None:
    gateway = _QuoteGateway(
        {"005930": Decimal(100_000), "000660": Decimal(1)}, Decimal(1)
    )

    outcome = _paper_cycle(tmp_path, gateway, [_buy()], session="closed").run()

    assert outcome.fills[0].status is FillStatus.UNFILLED
    assert outcome.portfolio.cash == Decimal(10_000_000)


def test_skipped_cycle_records_no_valuation(tmp_path: Path) -> None:
    gateway = _QuoteGateway({"005930": Decimal(1), "000660": Decimal(1)}, Decimal(1))

    outcome = _paper_cycle(tmp_path, gateway, [], status=CycleStatus.SKIPPED).run()

    assert outcome.valuation is None
    history = SQLitePaperPortfolioStore(tmp_path / "paper.db").history()
    assert history.valuations == []
    assert history.status_counts == {"skipped": 1}


def test_report_computes_returns_drawdown_and_benchmarks(tmp_path: Path) -> None:
    prices = {"005930": Decimal(100_000), "000660": Decimal(200_000)}
    gateway = _QuoteGateway(prices, Decimal(2_500))
    _paper_cycle(tmp_path, gateway, [_buy(quantity=20)]).run()
    store = SQLitePaperPortfolioStore(tmp_path / "paper.db")
    mark = MarkPaperPortfolio(
        store, gateway, ["005930", "000660"], Decimal(10_000_000), _Clock()
    )
    gateway.prices = {"005930": Decimal(90_000), "000660": Decimal(220_000)}
    gateway.kospi = Decimal(2_550)
    mark.run()
    gateway.prices = {"005930": Decimal(110_000), "000660": Decimal(220_000)}
    mark.run()

    portfolio = store.load()
    assert portfolio is not None
    report = build_paper_report(portfolio, store.history(), ["005930", "000660"])

    cash = Decimal(10_000_000 - 2_000_000 - 300)
    assert report.cash == cash
    assert report.total_value == cash + Decimal(2_200_000)
    assert report.total_return_percent == pytest.approx(
        (cash + 2_200_000) / Decimal(10_000_000) * 100 - 100
    )
    assert report.unrealized_pnl == Decimal(200_000)
    assert report.fees_and_taxes == Decimal(300)
    assert report.kospi_return_percent == pytest.approx(Decimal(2))
    # (1.1 + 1.1) / 2 - 1 = 10%
    assert report.watchlist_return_percent == pytest.approx(Decimal(10))
    # 고점은 초기 자금 10,000,000부터 시작: 10,000,000 -> 9,799,700
    assert report.max_drawdown_percent == pytest.approx(Decimal("-2.003"))
    assert len(report.daily) == 1
    assert report.holdings[0].weight_percent == pytest.approx(
        Decimal(2_200_000) / report.total_value * 100
    )
