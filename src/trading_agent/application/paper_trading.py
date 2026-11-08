"""가상 계좌로 판단 사이클을 돌리고 승인 제안을 모의 체결해 수익률을 추적한다."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

from trading_agent.application.ports import (
    DailyBar,
    IndexPoint,
    IndicatorPoint,
    MarketDataGateway,
    MarketIndex,
    NewsItem,
    StockQuote,
    TreasuryMaturity,
)
from trading_agent.application.records import CycleStatus, DecisionCycleResult
from trading_agent.domain.models import AccountSnapshot
from trading_agent.domain.paper import (
    FeeSchedule,
    FillStatus,
    PaperFill,
    PaperPortfolio,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Seoul")
_EXECUTABLE_STATUSES = (CycleStatus.COMPLETED, CycleStatus.MODEL_FAILED)


@dataclass(frozen=True, slots=True)
class PaperValuation:
    """한 시점의 가상 계좌 평가와 비교 기준 가격."""

    at: datetime
    cash: Decimal
    holdings_value: Decimal
    kospi: Decimal | None
    prices: dict[str, Decimal] = field(default_factory=dict)
    stale_codes: tuple[str, ...] = ()

    @property
    def total_value(self) -> Decimal:
        """현금과 보유 종목 평가액의 합."""

        return self.cash + self.holdings_value


@dataclass(frozen=True, slots=True)
class PaperRecord:
    """판단 사이클 또는 평가만 한 실행의 저장 단위."""

    kind: str
    started_at: datetime
    finished_at: datetime
    status: str
    run_id: int | None = None
    error: str | None = None
    valuation: PaperValuation | None = None
    fills: tuple[PaperFill, ...] = ()


@dataclass(frozen=True, slots=True)
class PaperFillRow:
    """저장된 가상 체결 이력 한 줄."""

    at: datetime
    code: str
    name: str
    action: str
    status: str
    requested_quantity: int
    quantity: int
    limit_price: Decimal
    fill_price: Decimal | None
    commission: Decimal
    tax: Decimal
    realized_pnl: Decimal | None
    source: str
    note: str | None


@dataclass(frozen=True, slots=True)
class PaperValuationRow:
    """저장된 평가 이력 한 줄."""

    at: datetime
    kind: str
    status: str
    cash: Decimal
    holdings_value: Decimal
    total_value: Decimal
    kospi: Decimal | None
    prices: dict[str, Decimal]


@dataclass(frozen=True, slots=True)
class PaperHistory:
    """리포트 계산에 필요한 저장 이력."""

    started_at: datetime
    valuations: list[PaperValuationRow]
    fills: list[PaperFillRow]
    status_counts: dict[str, int]


class PaperPortfolioStore(ABC):
    """가상 계좌 상태와 실행 이력 저장 포트."""

    @abstractmethod
    def load(self) -> PaperPortfolio | None:
        """저장된 가상 계좌를 읽는다. 없으면 None."""

    @abstractmethod
    def create(self, portfolio: PaperPortfolio, at: datetime) -> None:
        """새 가상 계좌를 만든다. 이미 있으면 실패한다."""

    @abstractmethod
    def save(self, portfolio: PaperPortfolio, record: PaperRecord) -> None:
        """계좌 상태와 실행 기록을 한 트랜잭션으로 저장한다."""

    @abstractmethod
    def history(self) -> PaperHistory:
        """리포트용 이력을 읽는다."""


class PaperAccountGateway(MarketDataGateway):
    """계좌 조회만 가상 계좌로 바꾸고 나머지는 실제 시장 데이터를 쓴다."""

    def __init__(self, delegate: MarketDataGateway, portfolio: PaperPortfolio) -> None:
        self.delegate = delegate
        self.portfolio = portfolio

    def get_account(self) -> AccountSnapshot:
        return self.portfolio.account_snapshot()

    def get_stock_quote(self, code: str) -> StockQuote:
        return self.delegate.get_stock_quote(code)

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        return self.delegate.get_daily_bars(code, start_date, end_date)

    def get_moving_average(
        self, code: str, start_date: str, end_date: str, window: int
    ) -> list[IndicatorPoint]:
        return self.delegate.get_moving_average(code, start_date, end_date, window)

    def get_rsi(
        self, code: str, start_date: str, end_date: str, window: int = 14
    ) -> list[IndicatorPoint]:
        return self.delegate.get_rsi(code, start_date, end_date, window)

    def get_rsi_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        rsi_window: int = 14,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        return self.delegate.get_rsi_signal(
            code, start_date, end_date, rsi_window, ema_window
        )

    def get_macd(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
    ) -> list[IndicatorPoint]:
        return self.delegate.get_macd(
            code, start_date, end_date, short_window, long_window
        )

    def get_macd_signal(
        self,
        code: str,
        start_date: str,
        end_date: str,
        short_window: int = 12,
        long_window: int = 26,
        ema_window: int = 9,
    ) -> list[IndicatorPoint]:
        return self.delegate.get_macd_signal(
            code, start_date, end_date, short_window, long_window, ema_window
        )

    def get_stock_news(self, code: str, search_date: str) -> list[NewsItem]:
        return self.delegate.get_stock_news(code, search_date)

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        return self.delegate.get_market_index(index, start_date, end_date)

    def get_fear_and_greed_index(self) -> IndexPoint:
        return self.delegate.get_fear_and_greed_index()

    def get_treasury_yield(
        self, maturity: TreasuryMaturity, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        return self.delegate.get_treasury_yield(maturity, start_date, end_date)


class _Cycle(Protocol):
    """RunDecisionCycle과 같은 실행 인터페이스."""

    def run(self) -> DecisionCycleResult: ...


@dataclass(frozen=True, slots=True)
class PaperCycleOutcome:
    """모의 사이클의 판단 결과, 체결, 평가."""

    result: DecisionCycleResult
    fills: tuple[PaperFill, ...]
    valuation: PaperValuation | None
    portfolio: PaperPortfolio


class _PaperBase:
    def __init__(
        self,
        store: PaperPortfolioStore,
        gateway: MarketDataGateway,
        watchlist: Sequence[str],
        initial_cash: Decimal,
        clock: Callable[[], datetime],
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.watchlist = tuple(watchlist)
        self.initial_cash = initial_cash
        self.clock = clock

    def _portfolio(self, at: datetime) -> PaperPortfolio:
        portfolio = self.store.load()
        if portfolio is None:
            portfolio = PaperPortfolio(self.initial_cash, self.initial_cash)
            self.store.create(portfolio, at)
        return portfolio

    def _value(
        self, portfolio: PaperPortfolio, known: dict[str, Decimal]
    ) -> PaperValuation:
        """보유·관심 종목 현재가와 KOSPI로 평가한다. 조회 실패 종목은 평균가로 둔다."""

        prices = dict(known)
        for code in dict.fromkeys([*portfolio.holdings, *self.watchlist]):
            if code in prices:
                continue
            try:
                prices[code] = self.gateway.get_stock_quote(code).current_price
            except Exception:
                continue
        stale = tuple(code for code in portfolio.holdings if code not in prices)
        at = self.clock()
        return PaperValuation(
            at=at,
            cash=portfolio.cash,
            holdings_value=portfolio.market_value(prices),
            kospi=self._kospi(at),
            prices=prices,
            stale_codes=stale,
        )

    def _kospi(self, at: datetime) -> Decimal | None:
        local = at.astimezone(MARKET_TIMEZONE).date()
        try:
            points = self.gateway.get_market_index(
                MarketIndex.KOSPI,
                (local - timedelta(days=14)).strftime("%Y%m%d"),
                local.strftime("%Y%m%d"),
            )
        except Exception:
            return None
        if not points:
            return None
        return max(points, key=lambda point: point.date).value


class RunPaperCycle(_PaperBase):
    """가상 계좌로 판단 사이클을 실행하고 승인 제안을 현재가로 모의 체결한다."""

    def __init__(
        self,
        store: PaperPortfolioStore,
        gateway: MarketDataGateway,
        cycle_factory: Callable[[MarketDataGateway], _Cycle],
        fees: FeeSchedule,
        watchlist: Sequence[str],
        initial_cash: Decimal,
        clock: Callable[[], datetime],
    ) -> None:
        super().__init__(store, gateway, watchlist, initial_cash, clock)
        self.cycle_factory = cycle_factory
        self.fees = fees

    def run(self) -> PaperCycleOutcome:
        started_at = self.clock()
        portfolio = self._portfolio(started_at)
        result = self.cycle_factory(PaperAccountGateway(self.gateway, portfolio)).run()

        quotes: dict[str, Decimal] = {}
        fills: list[PaperFill] = []
        snapshot = result.snapshot
        market_open = snapshot is not None and snapshot.market_session == "open"
        stocks = snapshot.stocks if snapshot is not None else []
        names = {stock.code: stock.name for stock in stocks}
        approved = [verdict for verdict in result.verdicts if verdict.approved]
        if result.status in _EXECUTABLE_STATUSES:
            for verdict in approved:
                proposal = verdict.proposal
                name = names.get(proposal.code, proposal.code)
                if not market_open:
                    fills.append(
                        PaperFill(
                            proposal,
                            name,
                            FillStatus.UNFILLED,
                            None,
                            note="장 운영 시간 외라 미체결",
                        )
                    )
                    continue
                try:
                    quote = self.gateway.get_stock_quote(proposal.code)
                except Exception as error:
                    fills.append(
                        PaperFill(
                            proposal,
                            name,
                            FillStatus.UNFILLED,
                            None,
                            note=f"체결 시점 현재가 조회 실패: {type(error).__name__}",
                        )
                    )
                    continue
                quotes[proposal.code] = quote.current_price
                fills.append(
                    portfolio.execute(proposal, name, quote.current_price, self.fees)
                )

        valuation = None
        error = result.error
        if result.status is not CycleStatus.SKIPPED:
            try:
                valuation = self._value(portfolio, quotes)
            except Exception as value_error:
                detail = f"평가 실패: {type(value_error).__name__}: {value_error}"
                error = f"{error}; {detail}" if error else detail
        self.store.save(
            portfolio,
            PaperRecord(
                kind="cycle",
                started_at=started_at,
                finished_at=self.clock(),
                status=str(result.status),
                run_id=result.run_id,
                error=error,
                valuation=valuation,
                fills=tuple(fills),
            ),
        )
        return PaperCycleOutcome(result, tuple(fills), valuation, portfolio)


class MarkPaperPortfolio(_PaperBase):
    """매매 없이 현재가로 가상 계좌를 평가해 기록한다(장 마감 평가용)."""

    def run(self) -> PaperValuation:
        started_at = self.clock()
        portfolio = self._portfolio(started_at)
        valuation = self._value(portfolio, {})
        self.store.save(
            portfolio,
            PaperRecord(
                kind="mark",
                started_at=started_at,
                finished_at=self.clock(),
                status="completed",
                valuation=valuation,
            ),
        )
        return valuation


@dataclass(frozen=True, slots=True)
class HoldingReport:
    """리포트에 표시할 보유 종목 평가."""

    code: str
    name: str
    quantity: int
    average_price: Decimal
    price: Decimal | None
    market_value: Decimal
    unrealized_pnl: Decimal
    return_percent: Decimal
    weight_percent: Decimal


@dataclass(frozen=True, slots=True)
class DailyReport:
    """거래일별 마지막 평가."""

    date: str
    total_value: Decimal
    daily_return_percent: Decimal | None
    cumulative_return_percent: Decimal


@dataclass(frozen=True, slots=True)
class PaperReport:
    """가상 계좌 성과 요약."""

    started_at: datetime
    as_of: datetime | None
    initial_cash: Decimal
    cash: Decimal
    holdings_value: Decimal
    total_value: Decimal
    total_return_percent: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees_and_taxes: Decimal
    max_drawdown_percent: Decimal
    kospi_return_percent: Decimal | None
    watchlist_return_percent: Decimal | None
    holdings: list[HoldingReport]
    daily: list[DailyReport]
    recent_fills: list[PaperFillRow]
    filled_count: int
    unfilled_count: int
    status_counts: dict[str, int]
    stale_codes: tuple[str, ...]


def build_paper_report(
    portfolio: PaperPortfolio,
    history: PaperHistory,
    watchlist: Sequence[str],
    recent_fills: int = 10,
) -> PaperReport:
    """저장된 마지막 평가 가격 기준으로 성과 지표를 계산한다."""

    hundred = Decimal(100)
    valuations = history.valuations
    latest = valuations[-1] if valuations else None
    prices = latest.prices if latest else {}
    holdings_value = portfolio.market_value(prices)
    total_value = portfolio.cash + holdings_value

    holdings = []
    for code, holding in portfolio.holdings.items():
        price = prices.get(code)
        value = (price or holding.average_price) * holding.quantity
        cost = holding.average_price * holding.quantity
        holdings.append(
            HoldingReport(
                code=code,
                name=holding.name,
                quantity=holding.quantity,
                average_price=holding.average_price,
                price=price,
                market_value=value,
                unrealized_pnl=value - cost,
                return_percent=(value - cost) / cost * hundred,
                weight_percent=value / total_value * hundred,
            )
        )

    peak = portfolio.initial_cash
    max_drawdown = Decimal(0)
    for row in valuations:
        peak = max(peak, row.total_value)
        max_drawdown = min(max_drawdown, (row.total_value - peak) / peak * hundred)

    kospi_return = None
    kospi_points = [row.kospi for row in valuations if row.kospi is not None]
    if len(kospi_points) >= 2:
        kospi_return = (kospi_points[-1] / kospi_points[0] - 1) * hundred

    daily: list[DailyReport] = []
    by_date: dict[str, PaperValuationRow] = {}
    for row in valuations:
        by_date[row.at.astimezone(MARKET_TIMEZONE).strftime("%Y-%m-%d")] = row
    previous = portfolio.initial_cash
    for date, row in by_date.items():
        daily.append(
            DailyReport(
                date=date,
                total_value=row.total_value,
                daily_return_percent=(row.total_value / previous - 1) * hundred,
                cumulative_return_percent=(row.total_value / portfolio.initial_cash - 1)
                * hundred,
            )
        )
        previous = row.total_value

    filled = [row for row in history.fills if row.status == FillStatus.FILLED]
    return PaperReport(
        started_at=history.started_at,
        as_of=latest.at if latest else None,
        initial_cash=portfolio.initial_cash,
        cash=portfolio.cash,
        holdings_value=holdings_value,
        total_value=total_value,
        total_return_percent=(total_value / portfolio.initial_cash - 1) * hundred,
        realized_pnl=sum(
            (row.realized_pnl for row in filled if row.realized_pnl is not None),
            Decimal(0),
        ),
        unrealized_pnl=sum((item.unrealized_pnl for item in holdings), Decimal(0)),
        fees_and_taxes=sum((row.commission + row.tax for row in filled), Decimal(0)),
        max_drawdown_percent=max_drawdown,
        kospi_return_percent=kospi_return,
        watchlist_return_percent=_equal_weight_return(valuations, watchlist),
        holdings=holdings,
        daily=daily,
        recent_fills=history.fills[-recent_fills:],
        filled_count=len(filled),
        unfilled_count=len(history.fills) - len(filled),
        status_counts=history.status_counts,
        stale_codes=tuple(code for code in portfolio.holdings if code not in prices),
    )


def _equal_weight_return(
    valuations: Sequence[PaperValuationRow], watchlist: Sequence[str]
) -> Decimal | None:
    """첫 평가 시점에 관심 종목을 균등 매수해 보유했을 때의 수익률(비용 제외)."""

    if len(valuations) < 2 or not watchlist:
        return None
    first, last = valuations[0].prices, valuations[-1].prices
    codes = [code for code in watchlist if code in first and code in last]
    if not codes:
        return None
    ratios = [last[code] / first[code] for code in codes]
    return (sum(ratios, Decimal(0)) / len(ratios) - 1) * Decimal(100)
