"""저장된 과거 추천을 이후 가격으로 채점하고 오프라인 재채점한다."""

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from math import sqrt
from statistics import median
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from trading_agent.application.backtest import BACKTEST_FORMAT_VERSION
from trading_agent.application.ports import (
    DailyBar,
    IndexPoint,
    MarketIndex,
    StockQuote,
)
from trading_agent.domain.models import (
    Action,
    ConfidenceAggregation,
    ConsensusPolicyName,
)
from trading_agent.domain.recommendation import (
    ModelRecommendation,
    RecommendationConsensusPolicy,
    StockRecommendation,
)

DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_HOLD_BANDS = {5: 2.5, 10: 3.5, 20: 5.0}
CALIBRATION_BOUNDS = ((0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0))
SOURCE_NAMES = ("claude", "codex", "consensus")
SCORE_FORMAT_VERSION = 1
KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


class ScoreGateway(Protocol):
    """이후 가격과 시장 정보를 조회하는 실제 시장 데이터 포트."""

    def get_stock_quote(self, code: str) -> StockQuote: ...

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]: ...

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]: ...


class ScoreResultStore(Protocol):
    """백테스트 실행 결과를 조회하는 최소 포트."""

    def read_manifest(self) -> Mapping[str, Any] | None: ...

    def read_date_result(self, date: str) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class BacktestScoreOptions:
    """백테스트 채점과 재채점 옵션."""

    horizons: tuple[int, ...] | None = None
    policy: ConsensusPolicyName | None = None
    confidence_aggregation: ConfidenceAggregation | None = None
    compare_policies: bool = False
    bootstrap_iterations: int = 4000
    seed: int = 20261103
    hold_bands: Mapping[int, float] | None = None


@dataclass(frozen=True, slots=True)
class BacktestScoreResult:
    """JSON 직렬화 결과와 Markdown 요약."""

    data: dict[str, Any]
    markdown: str


@dataclass(frozen=True, slots=True)
class _LoadedDate:
    date: str
    result: Mapping[str, Any]
    models: Mapping[str, ModelRecommendation]


@dataclass(frozen=True, slots=True)
class _Observation:
    date: str
    code: str
    source: str
    horizon: int
    action: Action | None
    confidence: float | None
    xs: float | None
    exc: float | None
    close: Decimal | None
    target: Decimal | None
    invalidation: Decimal | None
    disagreement: bool = False


class ScoreBacktest:
    """결과 폴더를 읽고 가격 조회, 재합의, 지표 계산을 수행한다."""

    def __init__(
        self,
        result_store: ScoreResultStore,
        gateway: ScoreGateway,
        options: BacktestScoreOptions,
        now: datetime | None = None,
    ) -> None:
        self.result_store = result_store
        self.gateway = gateway
        self.options = options
        self.now = now or datetime.now(KOREA_TIMEZONE)
        self.missing_prices = 0
        self.missing_benchmarks = 0
        self.quote_failures = 0
        self.bar_failures = 0
        self.index_failures = 0

    def run(self) -> BacktestScoreResult:
        """스키마를 확인하고 점수 JSON 구조와 Markdown 요약을 반환한다."""

        manifest = self.result_store.read_manifest()
        if manifest is None:
            raise ValueError("manifest.json을 찾을 수 없습니다.")
        if manifest.get("format_version") != BACKTEST_FORMAT_VERSION:
            raise ValueError("지원하지 않는 백테스트 format_version입니다.")
        if self.options.bootstrap_iterations <= 0:
            raise ValueError("bootstrap 반복 수는 양수여야 합니다.")

        date_names = manifest.get("selected_dates")
        if not isinstance(date_names, list) or not all(
            isinstance(item, str) for item in date_names
        ):
            raise ValueError("manifest selected_dates가 올바르지 않습니다.")
        arguments = manifest.get("arguments", {})
        if not isinstance(arguments, Mapping):
            arguments = {}
        saved_horizons = arguments.get("horizons", DEFAULT_HORIZONS)
        horizons = (
            _positive_horizons(self.options.horizons)
            if self.options.horizons is not None
            else _positive_horizons(saved_horizons)
        )
        if not horizons:
            raise ValueError("horizons가 비어 있습니다.")
        policy = ConsensusPolicyName(
            self.options.policy
            or _mapping_value(manifest.get("consensus"), "policy", "strict")
        )
        aggregation = ConfidenceAggregation(
            self.options.confidence_aggregation
            or _mapping_value(
                manifest.get("consensus"), "confidence_aggregation", "min"
            )
        )
        hold_bands = {**DEFAULT_HOLD_BANDS, **(self.options.hold_bands or {})}

        dates, status_counts, source_failures, missing_items, leakage = (
            self._load_dates(date_names)
        )
        codes = self._codes(arguments, dates)
        prices, benchmark_values, market_by_code = self._load_prices(
            codes, dates, max(horizons)
        )
        xs_returns, exc_returns, absolute_returns = _returns_by_horizon(
            dates, codes, horizons, prices, benchmark_values, market_by_code
        )
        self.missing_prices = sum(value is None for value in absolute_returns.values())
        self.missing_benchmarks = sum(value is None for value in exc_returns.values())
        all_rows = self._observations(
            dates,
            codes,
            horizons,
            xs_returns,
            exc_returns,
            prices,
            hold_bands,
            policy,
            aggregation,
        )
        periods = {
            "before_20260701": [item for item in dates if item.date < "20260701"],
            "from_20260701": [item for item in dates if item.date >= "20260701"],
        }
        selected_policy = _policy_result(
            all_rows,
            dates,
            horizons,
            absolute_returns,
            hold_bands,
            self.options.bootstrap_iterations,
            self.options.seed,
            "selected",
        )
        period_results = {
            name: _policy_result(
                [row for row in all_rows if row.date in {item.date for item in subset}],
                subset,
                horizons,
                absolute_returns,
                hold_bands,
                self.options.bootstrap_iterations,
                self.options.seed,
                name,
            )
            for name, subset in periods.items()
        }
        comparisons = self._policy_comparisons(
            dates,
            codes,
            horizons,
            xs_returns,
            exc_returns,
            prices,
            hold_bands,
            policy,
            aggregation,
            absolute_returns,
        )

        selected_dates = len(date_names)
        warning = any(date < "20260701" for date in date_names)
        result_data: dict[str, Any] = {
            "format_version": SCORE_FORMAT_VERSION,
            "source_format_version": BACKTEST_FORMAT_VERSION,
            "mode": manifest.get("mode", "unknown"),
            "horizons": list(horizons),
            "policy": policy.value,
            "confidence_aggregation": aggregation.value,
            "hold_bands_xs_percent": {
                str(horizon): hold_bands.get(horizon, DEFAULT_HOLD_BANDS.get(horizon))
                for horizon in horizons
            },
            "bootstrap_iterations": self.options.bootstrap_iterations,
            "seed": self.options.seed,
            "counts": {
                "selected_dates": selected_dates,
                "completed_dates": len(dates),
                "excluded_dates": selected_dates - len(dates),
                "statuses": dict(status_counts),
                "source_failures": source_failures,
                "missing_snapshot_items": missing_items,
                "missing_future_prices": self.missing_prices,
                "missing_benchmarks": self.missing_benchmarks,
                "quote_failures": self.quote_failures,
                "bar_failures": self.bar_failures,
                "index_failures": self.index_failures,
                "leakage_blocked_rows": leakage,
            },
            "memory_leak_warning": warning,
            "sources": selected_policy["sources"],
            "base_rates": selected_policy["base_rates"],
            "price_outcomes": self._price_outcomes(
                dates, codes, all_rows, prices, max(horizons)
            ),
            "disagreements": selected_policy["disagreements"],
            "policy_comparison": comparisons,
            "periods": period_results,
            "generated_at": self.now.isoformat(),
        }
        return BacktestScoreResult(result_data, render_score_markdown(result_data))

    def _load_dates(
        self, date_names: Sequence[str]
    ) -> tuple[list[_LoadedDate], Counter[str], dict[str, int], int, int]:
        completed: list[_LoadedDate] = []
        statuses: Counter[str] = Counter()
        model_failures = Counter()
        missing_items = 0
        leakage_rows = 0
        for date in date_names:
            result = self.result_store.read_date_result(date)
            if result is None:
                statuses["missing_result"] += 1
                continue
            if result.get("format_version") != BACKTEST_FORMAT_VERSION:
                raise ValueError(f"{date} 결과의 format_version이 지원되지 않습니다.")
            status = str(result.get("status", "unknown"))
            statuses[status] += 1
            missing = result.get("missing", [])
            missing_items += len(missing) if isinstance(missing, list) else 0
            leakage = result.get("leakage")
            if isinstance(leakage, Mapping):
                leakage_rows += sum(
                    int(value)
                    for value in leakage.values()
                    if isinstance(value, (int, float))
                )
            if status != "completed":
                continue
            models: dict[str, ModelRecommendation] = {}
            for index, item in enumerate(result.get("models", [])):
                if not isinstance(item, Mapping):
                    continue
                model_id = str(item.get("model_id", "")).lower()
                name = (
                    model_id
                    if model_id in ("claude", "codex")
                    else SOURCE_NAMES[min(index, 1)]
                )
                decision = item.get("restored_decision") or item.get("decision")
                if decision is None:
                    model_failures[name] += 1
                    continue
                models[name] = _model_recommendation(name, decision)
            if "claude" in models and "codex" in models:
                completed.append(_LoadedDate(date, result, models))
            else:
                statuses["completed_missing_model"] += 1
        return completed, statuses, dict(model_failures), missing_items, leakage_rows

    def _codes(
        self, arguments: Mapping[str, Any], dates: Sequence[_LoadedDate]
    ) -> list[str]:
        raw_codes = arguments.get("universe", [])
        codes = [str(item) for item in raw_codes] if isinstance(raw_codes, list) else []
        for item in dates:
            for model in item.models.values():
                codes.extend(rec.code for rec in model.recommendations)
        return list(dict.fromkeys(codes))

    def _load_prices(
        self, codes: Sequence[str], dates: Sequence[_LoadedDate], max_horizon: int
    ) -> tuple[
        dict[tuple[str, str], list[DailyBar]],
        dict[tuple[MarketIndex, str], Decimal],
        dict[str, MarketIndex | None],
    ]:
        if not dates or not codes:
            return {}, {}, {}
        first = min(item.date for item in dates)
        last = max(item.date for item in dates)
        start = (datetime.strptime(first, "%Y%m%d") - timedelta(days=2)).strftime(
            "%Y%m%d"
        )
        calculated_end = (
            datetime.strptime(last, "%Y%m%d") + timedelta(days=max_horizon * 4 + 14)
        ).strftime("%Y%m%d")
        if self.now.tzinfo is None or self.now.utcoffset() is None:
            raise ValueError("score now는 timezone-aware datetime이어야 합니다.")
        last_complete_date = (
            self.now.astimezone(KOREA_TIMEZONE).date() - timedelta(days=1)
        ).strftime("%Y%m%d")
        end = min(calculated_end, last_complete_date)
        price_bars: dict[tuple[str, str], list[DailyBar]] = {}
        markets: dict[str, MarketIndex | None] = {}
        for code in codes:
            try:
                quote = self.gateway.get_stock_quote(code)
            except Exception:
                self.quote_failures += 1
                quote = None
            markets[code] = market_index_for(quote.market_name) if quote else None
            if start <= end:
                try:
                    bars = sorted(
                        self.gateway.get_daily_bars(code, start, end),
                        key=lambda bar: bar.date,
                    )
                except Exception:
                    self.bar_failures += 1
                    bars = []
            else:
                bars = []
            price_bars[(code, "all")] = bars

        if start > end:
            return price_bars, {}, markets
        indexes = {market for market in markets.values() if market is not None}
        benchmark_values: dict[tuple[MarketIndex, str], Decimal] = {}
        for market in sorted(indexes, key=lambda item: item.value):
            try:
                points = self.gateway.get_market_index(market, start, end)
            except Exception:
                self.index_failures += 1
                points = []
            benchmark_values.update(
                ((market, point.date), point.value) for point in points
            )
        return price_bars, benchmark_values, markets

    def _observations(
        self,
        dates: Sequence[_LoadedDate],
        codes: Sequence[str],
        horizons: Sequence[int],
        xs_returns: Mapping[tuple[str, str, int], float | None],
        exc_returns: Mapping[tuple[str, str, int], float | None],
        prices: Mapping[tuple[str, str], list[DailyBar]],
        hold_bands: Mapping[int, float],
        policy: ConsensusPolicyName,
        aggregation: ConfidenceAggregation,
    ) -> list[_Observation]:
        rows: list[_Observation] = []
        for dated in dates:
            claude, codex = dated.models["claude"], dated.models["codex"]
            consensus = RecommendationConsensusPolicy.reconcile(
                claude, codex, policy, aggregation
            )
            recs_by_source: dict[str, dict[str, StockRecommendation]] = {
                "claude": _first_by_code(claude.recommendations),
                "codex": _first_by_code(codex.recommendations),
                "consensus": _first_by_code(consensus.recommendations),
            }
            disagreement_codes = {item.code for item in consensus.disagreements}
            for source in SOURCE_NAMES:
                source_recs = recs_by_source[source]
                for code in codes:
                    rec = source_recs.get(code)
                    disagreement = source == "consensus" and code in disagreement_codes
                    action = (
                        None if disagreement else (rec.rating if rec else Action.HOLD)
                    )
                    confidence = rec.confidence if rec else None
                    bars = prices.get((code, "all"), [])
                    decision_bar = next(
                        (bar for bar in bars if bar.date == dated.date), None
                    )
                    for horizon in horizons:
                        key = (dated.date, code, horizon)
                        rows.append(
                            _Observation(
                                date=dated.date,
                                code=code,
                                source=source,
                                horizon=horizon,
                                action=action,
                                confidence=confidence,
                                xs=xs_returns.get(key),
                                exc=exc_returns.get(key),
                                close=decision_bar.close_price
                                if decision_bar
                                else None,
                                target=rec.target_price if rec else None,
                                invalidation=rec.invalidation_price if rec else None,
                                disagreement=disagreement,
                            )
                        )
        return rows

    def _policy_comparisons(
        self,
        dates: Sequence[_LoadedDate],
        codes: Sequence[str],
        horizons: Sequence[int],
        xs_returns: Mapping[tuple[str, str, int], float | None],
        exc_returns: Mapping[tuple[str, str, int], float | None],
        prices: Mapping[tuple[str, str], list[DailyBar]],
        hold_bands: Mapping[int, float],
        policy: ConsensusPolicyName,
        aggregation: ConfidenceAggregation,
        absolute_returns: Mapping[tuple[str, str, int], float | None],
    ) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        policies = (
            list(ConsensusPolicyName) if self.options.compare_policies else [policy]
        )
        aggregations = (
            list(ConfidenceAggregation)
            if self.options.compare_policies
            else [aggregation]
        )
        for compare_policy in policies:
            for compare_aggregation in aggregations:
                rows: list[_Observation] = []
                for dated in dates:
                    merged = RecommendationConsensusPolicy.reconcile(
                        dated.models["claude"],
                        dated.models["codex"],
                        compare_policy,
                        compare_aggregation,
                    )
                    accepted = _first_by_code(merged.recommendations)
                    disagreements = {item.code for item in merged.disagreements}
                    for code in codes:
                        rec = accepted.get(code)
                        decision_bar = next(
                            (
                                bar
                                for bar in prices.get((code, "all"), [])
                                if bar.date == dated.date
                            ),
                            None,
                        )
                        for horizon in horizons:
                            key = (dated.date, code, horizon)
                            rows.append(
                                _Observation(
                                    date=dated.date,
                                    code=code,
                                    source="consensus",
                                    horizon=horizon,
                                    action=None
                                    if code in disagreements
                                    else (rec.rating if rec else Action.HOLD),
                                    confidence=rec.confidence if rec else None,
                                    xs=xs_returns.get(key),
                                    exc=exc_returns.get(key),
                                    close=decision_bar.close_price
                                    if decision_bar
                                    else None,
                                    target=rec.target_price if rec else None,
                                    invalidation=rec.invalidation_price
                                    if rec
                                    else None,
                                    disagreement=code in disagreements,
                                )
                            )
                result = _policy_result(
                    rows,
                    dates,
                    horizons,
                    absolute_returns,
                    hold_bands,
                    self.options.bootstrap_iterations,
                    self.options.seed,
                    f"{compare_policy.value}/{compare_aggregation.value}",
                )
                source_data = result["sources"]["consensus"]
                by_horizon = source_data["horizons"]
                comparisons.append(
                    {
                        "policy": compare_policy.value,
                        "confidence_aggregation": compare_aggregation.value,
                        "buy_n": source_data["opinion_distribution"].get("buy", 0),
                        "buy_xs10": by_horizon.get("10", {}).get("buy_mean_xs"),
                        "buy_xs10_ci95": by_horizon.get("10", {}).get("buy_ci95"),
                        "buy_xs20": by_horizon.get("20", {}).get("buy_mean_xs"),
                        "buy_xs20_ci95": by_horizon.get("20", {}).get("buy_ci95"),
                        "sell_xs20": by_horizon.get("20", {}).get("sell_mean_xs"),
                    }
                )
        return comparisons

    def _price_outcomes(
        self,
        dates: Sequence[_LoadedDate],
        codes: Sequence[str],
        rows: Sequence[_Observation],
        prices: Mapping[tuple[str, str], list[DailyBar]],
        max_horizon: int,
    ) -> dict[str, Any]:
        outcome_names = (
            "target_first",
            "invalidation_first",
            "both_same_day",
            "neither",
            "no_price",
        )
        counts: dict[str, dict[str, Counter[str]]] = {
            source: {
                rating: Counter({outcome: 0 for outcome in outcome_names})
                for rating in ("buy", "sell")
            }
            for source in SOURCE_NAMES
        }
        widths: dict[str, dict[str, dict[str, list[float]]]] = {
            source: {
                rating: {"target": [], "invalidation": []} for rating in ("buy", "sell")
            }
            for source in SOURCE_NAMES
        }
        date_by_name = {item.date: item for item in dates}
        for row in rows:
            if (
                row.horizon != max_horizon
                or row.action not in (Action.BUY, Action.SELL)
                or row.date not in date_by_name
            ):
                continue
            rating = row.action.value
            future_bars = [
                bar for bar in prices.get((row.code, "all"), []) if bar.date > row.date
            ][:max_horizon]
            if len(future_bars) < max_horizon:
                outcome = "no_price"
            else:
                outcome = price_outcome(
                    row.action, row.target, row.invalidation, future_bars
                )
            counts[row.source][rating][outcome] += 1
            if row.close is None or row.close == 0:
                continue
            if row.target is not None:
                widths[row.source][rating]["target"].append(
                    float((row.target - row.close) / row.close * 100)
                )
            if row.invalidation is not None:
                widths[row.source][rating]["invalidation"].append(
                    float((row.invalidation - row.close) / row.close * 100)
                )
        return {
            "by_source": {
                source: {
                    rating: {key: int(value) for key, value in outcome_counts.items()}
                    for rating, outcome_counts in source_counts.items()
                }
                for source, source_counts in counts.items()
            },
            "width_medians_percent": {
                source: {
                    rating: {
                        "target": _median(measures["target"]),
                        "invalidation": _median(measures["invalidation"]),
                    }
                    for rating, measures in source_widths.items()
                }
                for source, source_widths in widths.items()
            },
            "window_trading_days": max_horizon,
        }


def market_index_for(market_name: str | None) -> MarketIndex | None:
    """응답의 시장명으로 종목 벤치마크를 고른다."""

    normalized = (market_name or "").strip().upper()
    if (
        normalized.startswith("KOSPI")
        or "코스피" in normalized
        or "유가증권" in normalized
    ):
        return MarketIndex.KOSPI
    if normalized.startswith(("KOSDAQ", "KSQ")) or "코스닥" in normalized:
        return MarketIndex.KOSDAQ
    return None


def price_outcome(
    action: Action,
    target: Decimal | None,
    invalidation: Decimal | None,
    future_bars: Sequence[DailyBar],
) -> str:
    """최대 기간 안에서 목표가와 무효화 가격 중 먼저 닿은 결과를 정한다."""

    if target is None or invalidation is None or not future_bars:
        return "no_price"
    for bar in future_bars:
        if bar.high_price is None or bar.low_price is None:
            return "no_price"
        target_hit = invalidation_hit = False
        if action is Action.BUY:
            target_hit = target is not None and bar.high_price >= target
            invalidation_hit = (
                invalidation is not None and bar.low_price <= invalidation
            )
        elif action is Action.SELL:
            target_hit = target is not None and bar.low_price <= target
            invalidation_hit = (
                invalidation is not None and bar.high_price >= invalidation
            )
        if target_hit and invalidation_hit:
            return "both_same_day"
        if target_hit:
            return "target_first"
        if invalidation_hit:
            return "invalidation_first"
    return "neither"


def render_score_markdown(result: Mapping[str, Any]) -> str:
    """점수 JSON의 핵심 표와 해석 경고를 Markdown으로 만든다."""

    def row(*cells: Any) -> str:
        return "| " + " | ".join(str(cell) for cell in cells) + " |"

    horizons = [str(value) for value in result.get("horizons", [])]
    counts = result.get("counts", {})
    lines = [
        "# 백테스트 채점 요약",
        "",
        f"- 결과 형식: {result.get('mode', 'unknown')}",
        f"- 재채점 정책: {result.get('policy')} / "
        f"{result.get('confidence_aggregation')}",
        f"- 날짜: {counts.get('completed_dates', 0)} completed / "
        f"{counts.get('excluded_dates', 0)} 제외",
    ]
    if result.get("memory_leak_warning"):
        lines.extend(
            [
                "",
                "> 경고: 2026-07 이전 판단일은 모델 기억 누수 가능성이 있습니다. "
                "2026-07 이후 결과로 판단하세요.",
            ]
        )
    lines.extend(
        [
            "",
            "## 의견 분포",
            "",
            row("출처", "BUY", "HOLD", "SELL", "불일치"),
            row("---", "---:", "---:", "---:", "---:"),
        ]
    )
    for source, source_data in result.get("sources", {}).items():
        distribution = source_data.get("opinion_distribution", {})
        lines.append(
            row(
                source,
                distribution.get("buy", 0),
                distribution.get("hold", 0),
                distribution.get("sell", 0),
                distribution.get("disagreement", 0),
            )
        )
    lines.extend(
        [
            "",
            "## 방향 적중과 universe 초과수익(xs)",
            "",
            row(
                "출처",
                "horizon",
                "표본",
                "적중률",
                "buy xs 평균 [95%]",
                "sell xs 평균 [95%]",
            ),
            row("---", "---:", "---:", "---:", "---:", "---:"),
        ]
    )
    for source, source_data in result.get("sources", {}).items():
        for horizon in horizons:
            metric = source_data.get("horizons", {}).get(horizon, {})
            lines.append(
                row(
                    source,
                    horizon,
                    metric.get("n", 0),
                    _rate(metric.get("direction_hit_rate")),
                    _interval(metric.get("buy_mean_xs"), metric.get("buy_ci95")),
                    _interval(metric.get("sell_mean_xs"), metric.get("sell_ci95")),
                )
            )
    lines.extend(
        [
            "",
            "## 의견별 방향 적중",
            "",
            row(
                "출처",
                "horizon",
                "BUY n / 적중률",
                "HOLD n / 적중률",
                "SELL n / 적중률",
            ),
            row("---", "---:", "---:", "---:", "---:"),
        ]
    )
    for source, source_data in result.get("sources", {}).items():
        for horizon in horizons:
            metric = source_data.get("horizons", {}).get(horizon, {})
            ratings = metric.get("by_rating", {})
            lines.append(
                row(
                    source,
                    horizon,
                    _rating_hit(ratings.get("buy", {})),
                    _rating_hit(ratings.get("hold", {})),
                    _rating_hit(ratings.get("sell", {})),
                )
            )
    lines.extend(
        [
            "",
            "## 기준율",
            "",
            row(
                "horizon",
                "종목 수",
                "xs > 0",
                "xs hold band",
                "exc > 0",
                "절대수익 > 0",
            ),
            row("---:", "---:", "---:", "---:", "---:", "---:"),
        ]
    )
    for horizon, metric in result.get("base_rates", {}).items():
        lines.append(
            row(
                horizon,
                metric.get("universe_n", 0),
                _rate(metric.get("xs_positive_rate")),
                _rate(metric.get("xs_hold_band_rate")),
                _rate(metric.get("exc_positive_rate")),
                _rate(metric.get("absolute_positive_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## IC",
            "",
            row("출처", "horizon", "날짜 평균 Spearman", "t값", "날짜 수"),
            row("---", "---:", "---:", "---:", "---:"),
        ]
    )
    for source, source_data in result.get("sources", {}).items():
        for horizon in horizons:
            ic = source_data.get("horizons", {}).get(horizon, {}).get("ic", {})
            lines.append(
                row(
                    source,
                    horizon,
                    _number(ic.get("mean")),
                    _number(ic.get("t_value")),
                    ic.get("dates", 0),
                )
            )
    price = result.get("price_outcomes", {})
    lines.extend(
        [
            "",
            "## 목표가·무효화 가격 선착",
            "",
            row(
                "출처",
                "의견",
                "target_first",
                "invalidation_first",
                "both_same_day",
                "neither",
                "no_price",
                "손절가 변화 중앙값",
                "목표가 변화 중앙값",
            ),
            row("---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"),
        ]
    )
    outcome_names = (
        "target_first",
        "invalidation_first",
        "both_same_day",
        "neither",
        "no_price",
    )
    by_source = price.get("by_source", {})
    width_medians = price.get("width_medians_percent", {})
    for source in SOURCE_NAMES:
        for rating in ("buy", "sell"):
            outcomes = by_source.get(source, {}).get(rating, {})
            widths = width_medians.get(source, {}).get(rating, {})
            lines.append(
                row(
                    source,
                    rating,
                    *(outcomes.get(key, 0) for key in outcome_names),
                    _value_percent(widths.get("invalidation")),
                    _value_percent(widths.get("target")),
                )
            )
    lines.extend(
        [
            "",
            "## 확신도 보정",
            "",
            row("출처", "horizon", "의견군", "구간", "표본", "xs 적중률", "exc 적중률"),
            row("---", "---:", "---", "---:", "---:", "---:", "---:"),
        ]
    )
    for source, source_data in result.get("sources", {}).items():
        for horizon in horizons:
            calibration = (
                source_data.get("horizons", {}).get(horizon, {}).get("calibration", {})
            )
            for group in ("buy_sell", "hold"):
                for bucket in calibration.get(group, []):
                    lines.append(
                        row(
                            source,
                            horizon,
                            group,
                            bucket["bucket"],
                            bucket["n"],
                            _rate(bucket.get("hit_rate")),
                            _rate(bucket.get("exc_hit_rate")),
                        )
                    )
    lines.extend(
        [
            "",
            "## 불일치 유형",
            "",
            row("Claude/Codex", "건수", "평균 xs", "Claude 적중", "Codex 적중"),
            row("---", "---:", "---:", "---:", "---:"),
        ]
    )
    for kind, value in result.get("disagreements", {}).items():
        lines.append(
            row(
                kind,
                value.get("n", 0),
                _percentage_points(value.get("mean_xs")),
                _rate(value.get("claude_hit_rate")),
                _rate(value.get("codex_hit_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## 정책 비교",
            "",
            row(
                "policy",
                "confidence",
                "buy n",
                "buy xs10 [95%]",
                "buy xs20 [95%]",
                "sell xs20",
            ),
            row("---", "---", "---:", "---:", "---:", "---:"),
        ]
    )
    for item in result.get("policy_comparison", []):
        lines.append(
            row(
                item["policy"],
                item["confidence_aggregation"],
                item["buy_n"],
                _interval(item.get("buy_xs10"), item.get("buy_xs10_ci95")),
                _interval(item.get("buy_xs20"), item.get("buy_xs20_ci95")),
                _percentage_points(item.get("sell_xs20")),
            )
        )
    before = result.get("periods", {}).get("before_20260701", {})
    after = result.get("periods", {}).get("from_20260701", {})
    lines.extend(
        [
            "",
            "## 기간별 합의 성과",
            "",
            row(
                "기간",
                "horizon",
                "BUY n",
                "BUY 적중률",
                "BUY 평균 xs",
                "SELL n",
                "SELL 적중률",
                "SELL 평균 xs",
            ),
            row("---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"),
        ]
    )
    for period_name, period_data in (
        ("2026-07 이전", before),
        ("2026-07 이후", after),
    ):
        consensus = period_data.get("sources", {}).get("consensus", {})
        for horizon in horizons:
            metric = consensus.get("horizons", {}).get(horizon, {})
            ratings = metric.get("by_rating", {})
            buy = ratings.get("buy", {})
            sell = ratings.get("sell", {})
            lines.append(
                row(
                    period_name,
                    horizon,
                    metric.get("buy_n", 0),
                    _rate(buy.get("hit_rate")),
                    _percentage_points(metric.get("buy_mean_xs")),
                    metric.get("sell_n", 0),
                    _rate(sell.get("hit_rate")),
                    _percentage_points(metric.get("sell_mean_xs")),
                )
            )
    lines.extend(
        [
            "",
            "## 누락·실패와 기간 분할",
            "",
            f"- 누락 미래 가격: {counts.get('missing_future_prices', 0)}",
            f"- 누락 벤치마크: {counts.get('missing_benchmarks', 0)}",
            f"- quote/일봉/지수 실패: {counts.get('quote_failures', 0)} / "
            f"{counts.get('bar_failures', 0)} / "
            f"{counts.get('index_failures', 0)}",
            f"- 비완료 날짜 상태: `{counts.get('statuses', {})}`",
            f"- 누수 차단 합계: {counts.get('leakage_blocked_rows', 0)}",
            f"- 기간 분할: 이전 {before.get('date_count', 0)}일 / "
            f"이후 {after.get('date_count', 0)}일",
            "",
            "## 해석 주의",
            "",
            "xs는 같은 판단일 universe 평균 수익률을 뺀 상대수익입니다. 표본 크기와 "
            "누락 가격을 확인하고, 신뢰구간과 기간 분할을 함께 해석하세요.",
            "",
        ]
    )
    return "\n".join(lines)


def _policy_result(
    rows: Sequence[_Observation],
    dates: Sequence[_LoadedDate],
    horizons: Sequence[int],
    absolute_returns: Mapping[tuple[str, str, int], float | None],
    hold_bands: Mapping[int, float],
    iterations: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    source_rows = {
        source: [row for row in rows if row.source == source] for source in SOURCE_NAMES
    }
    sources: dict[str, Any] = {}
    for source in SOURCE_NAMES:
        observations = source_rows[source]
        unique = {
            (row.date, row.code): row
            for row in observations
            if row.horizon == max(horizons)
        }
        distributions = Counter()
        for row in unique.values():
            distributions[
                "disagreement"
                if row.disagreement
                else (row.action.value if row.action else "disagreement")
            ] += 1
        horizon_metrics = {}
        for horizon in horizons:
            selected = [
                row
                for row in observations
                if row.horizon == horizon and row.xs is not None
            ]
            actionable = [row for row in selected if row.action is not None]
            buys = [row for row in actionable if row.action is Action.BUY]
            sells = [row for row in actionable if row.action is Action.SELL]
            hits = [
                _is_hit(
                    row, hold_bands.get(horizon, DEFAULT_HOLD_BANDS.get(horizon, 0))
                )
                for row in actionable
            ]
            hits = [value for value in hits if value is not None]
            band = hold_bands.get(horizon, DEFAULT_HOLD_BANDS.get(horizon, 0))
            ic = _date_ic(selected)
            ratings: dict[str, Any] = {}
            for rating, group in (
                ("buy", buys),
                ("hold", [row for row in actionable if row.action is Action.HOLD]),
                ("sell", sells),
            ):
                rating_hits = [_is_hit(row, band) for row in group]
                rating_hits = [value for value in rating_hits if value is not None]
                ratings[rating] = {
                    "n": len(group),
                    "scored_n": len(rating_hits),
                    "hit_n": sum(rating_hits),
                    "hit_rate": _mean(rating_hits),
                }
            calibration_buy_sell = [
                row for row in actionable if row.action in (Action.BUY, Action.SELL)
            ]
            calibration_hold = [row for row in actionable if row.action is Action.HOLD]
            horizon_metrics[str(horizon)] = {
                "n": len(actionable),
                "direction_scored_n": len(hits),
                "direction_hit_n": sum(hits),
                "direction_hit_rate": _mean(hits),
                "buy_n": len(buys),
                "buy_mean_xs": _mean([row.xs for row in buys]),
                "buy_ci95": _cluster_bootstrap(buys, iterations, seed),
                "sell_n": len(sells),
                "sell_mean_xs": _mean([row.xs for row in sells]),
                "sell_ci95": _cluster_bootstrap(sells, iterations, seed + 1),
                "hold_n": sum(row.action is Action.HOLD for row in actionable),
                "by_rating": ratings,
                "ic": ic,
                "calibration": {
                    "buy_sell": _calibration(calibration_buy_sell, band),
                    "hold": _calibration(calibration_hold, band),
                },
            }
        sources[source] = {
            "opinion_distribution": {
                "buy": distributions.get("buy", 0),
                "hold": distributions.get("hold", 0),
                "sell": distributions.get("sell", 0),
                "disagreement": distributions.get("disagreement", 0),
            },
            "horizons": horizon_metrics,
        }
    base_rates = {}
    for horizon in horizons:
        universe_rows = [
            row
            for row in source_rows["claude"]
            if row.horizon == horizon and row.xs is not None
        ]
        band = hold_bands.get(horizon, DEFAULT_HOLD_BANDS.get(horizon, 0))
        base_rates[str(horizon)] = {
            "universe_n": len(universe_rows),
            "xs_positive_rate": _mean(
                [row.xs > 0 for row in universe_rows if row.xs is not None]
            ),
            "xs_hold_band_rate": _mean(
                [abs(row.xs) < band for row in universe_rows if row.xs is not None]
            ),
            "exc_positive_rate": _mean(
                [row.exc > 0 for row in universe_rows if row.exc is not None]
            ),
            "absolute_positive_rate": _mean(
                [
                    value > 0
                    for row in universe_rows
                    if (value := absolute_returns.get((row.date, row.code, horizon)))
                    is not None
                ]
            ),
        }
    disagreement_summary = _disagreement_summary(source_rows, max(horizons), hold_bands)
    return {
        "label": label,
        "date_count": len({item.date for item in dates}),
        "sources": sources,
        "base_rates": base_rates,
        "disagreements": disagreement_summary,
    }


def _returns_by_horizon(
    dates: Sequence[_LoadedDate],
    codes: Sequence[str],
    horizons: Sequence[int],
    prices: Mapping[tuple[str, str], list[DailyBar]],
    benchmarks: Mapping[tuple[MarketIndex, str], Decimal],
    markets: Mapping[str, MarketIndex | None],
) -> tuple[
    dict[tuple[str, str, int], float | None],
    dict[tuple[str, str, int], float | None],
    dict[tuple[str, str, int], float | None],
]:
    raw: dict[tuple[str, str, int], float | None] = {}
    for dated in dates:
        for code in codes:
            bars = prices.get((code, "all"), [])
            decision = next((bar for bar in bars if bar.date == dated.date), None)
            future = [bar for bar in bars if bar.date > dated.date]
            for horizon in horizons:
                key = (dated.date, code, horizon)
                if (
                    decision is None
                    or len(future) < horizon
                    or decision.close_price == 0
                ):
                    raw[key] = None
                    continue
                end_bar = future[horizon - 1]
                raw[key] = float((end_bar.close_price / decision.close_price - 1) * 100)
    # Universe mean includes every priced member, whether or not it was recommended.
    xs: dict[tuple[str, str, int], float | None] = {}
    for dated in dates:
        for horizon in horizons:
            values = [raw.get((dated.date, code, horizon)) for code in codes]
            available = [value for value in values if value is not None]
            average = _mean(available)
            for code in codes:
                value = raw.get((dated.date, code, horizon))
                xs[(dated.date, code, horizon)] = (
                    value - average
                    if value is not None and average is not None
                    else None
                )
    benchmark_returns: dict[tuple[str, str, int], float | None] = {}
    for dated in dates:
        for code in codes:
            index = markets.get(code)
            bars = prices.get((code, "all"), [])
            start = next((bar for bar in bars if bar.date == dated.date), None)
            future = [bar for bar in bars if bar.date > dated.date]
            for horizon in horizons:
                key = (dated.date, code, horizon)
                end = future[horizon - 1] if len(future) >= horizon else None
                base_value = benchmarks.get((index, dated.date)) if index else None
                end_value = benchmarks.get((index, end.date)) if index and end else None
                if (
                    start is None
                    or end is None
                    or base_value is None
                    or end_value is None
                    or base_value == 0
                ):
                    benchmark_returns[key] = None
                else:
                    benchmark_returns[key] = float((end_value / base_value - 1) * 100)
    exc = {
        key: (
            stock_return - benchmark_returns[key]
            if stock_return is not None and benchmark_returns.get(key) is not None
            else None
        )
        for key, stock_return in raw.items()
    }
    return xs, exc, raw


def _model_recommendation(name: str, value: Any) -> ModelRecommendation:
    if not isinstance(value, Mapping):
        raise ValueError("저장된 모델 추천이 object가 아닙니다.")
    items = value.get("recommendations", [])
    recommendations = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        recommendations.append(
            StockRecommendation(
                code=str(item["code"]),
                rating=Action(item["rating"]),
                confidence=float(item["confidence"]),
                target_price=_decimal_or_none(item.get("target_price")),
                invalidation_price=_decimal_or_none(item.get("invalidation_price")),
                reason=str(item.get("reason", "")),
                consensus_basis=item.get("consensus_basis"),
            )
        )
    return ModelRecommendation(
        model_id=name,
        recommendations=recommendations,
        summary=str(value.get("summary", "")),
        missing_info=list(value.get("missing_info", [])),
    )


def _first_by_code(
    items: Sequence[StockRecommendation],
) -> dict[str, StockRecommendation]:
    return {item.code: item for item in reversed(items)}


def _date_ic(
    observations: Sequence[_Observation],
) -> dict[str, Any]:
    grouped: dict[str, list[_Observation]] = defaultdict(list)
    for row in observations:
        grouped[row.date].append(row)
    values: list[float] = []
    for _day, rows in grouped.items():
        x = [
            0.0
            if row.action is None
            else _signed_score(row.action, row.confidence or 0.0)
            for row in rows
        ]
        y = [row.xs for row in rows]
        paired = [(a, b) for a, b in zip(x, y, strict=True) if b is not None]
        corr = _spearman([a for a, _ in paired], [b for _, b in paired])
        if corr is not None:
            values.append(corr)
    return {"mean": _mean(values), "t_value": _t_value(values), "dates": len(values)}


def _disagreement_summary(
    source_rows: Mapping[str, Sequence[_Observation]],
    horizon: int,
    hold_bands: Mapping[int, float],
) -> dict[str, Any]:
    by_type: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"n": [], "xs": [], "claude": [], "codex": []}
    )
    claude = {
        (row.date, row.code): row
        for row in source_rows["claude"]
        if row.horizon == horizon
    }
    codex = {
        (row.date, row.code): row
        for row in source_rows["codex"]
        if row.horizon == horizon
    }
    for row in source_rows["consensus"]:
        if not row.disagreement or row.horizon != horizon:
            continue
        left = claude.get((row.date, row.code))
        right = codex.get((row.date, row.code))
        key = (
            f"{left.action.value if left and left.action else 'hold'}/"
            f"{right.action.value if right and right.action else 'hold'}"
        )
        group = by_type[key]
        group["n"].append(1)
        if row.xs is not None:
            group["xs"].append(row.xs)
            for label, point in (("claude", left), ("codex", right)):
                if point and point.action is not None:
                    hit = _is_hit(
                        point,
                        hold_bands.get(horizon, DEFAULT_HOLD_BANDS.get(horizon, 0)),
                    )
                    if hit is not None:
                        group[label].append(hit)
    return {
        key: {
            "n": len(values["n"]),
            "mean_xs": _mean(values["xs"]),
            "claude_hit_rate": _mean(values["claude"]),
            "codex_hit_rate": _mean(values["codex"]),
        }
        for key, values in sorted(by_type.items())
    }


def _calibration(
    rows: Sequence[_Observation], hold_band: float
) -> list[dict[str, Any]]:
    output = []
    for index, (lower, upper) in enumerate(CALIBRATION_BOUNDS):
        selected = [
            row
            for row in rows
            if row.confidence is not None
            and row.confidence >= lower
            and (
                row.confidence <= upper
                if index == len(CALIBRATION_BOUNDS) - 1
                else row.confidence < upper
            )
            and row.xs is not None
        ]
        hits = [_is_hit(row, hold_band) for row in selected]
        hits = [value for value in hits if value is not None]
        exc_hits = [_is_hit_value(row.action, row.exc, hold_band) for row in selected]
        exc_hits = [value for value in exc_hits if value is not None]
        output.append(
            {
                "bucket": f"{lower:.1f}-{upper:.1f}",
                "n": len(hits),
                "hit_rate": _mean(hits),
                "exc_n": len(exc_hits),
                "exc_hit_rate": _mean(exc_hits),
            }
        )
    return output


def _is_hit(row: _Observation, hold_band: float) -> bool | None:
    return _is_hit_value(row.action, row.xs, hold_band)


def _is_hit_value(
    action: Action | None, value: float | None, hold_band: float
) -> bool | None:
    if action is None or value is None:
        return None
    if action is Action.BUY:
        return value > 0
    if action is Action.SELL:
        return value < 0
    return abs(value) < hold_band


def _signed_score(action: Action, confidence: float) -> float:
    if action is Action.BUY:
        return confidence
    if action is Action.SELL:
        return -confidence
    return 0.0


def _cluster_bootstrap(
    rows: Sequence[_Observation], iterations: int, seed: int
) -> list[float] | None:
    if not rows:
        return None
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.xs is not None:
            grouped[row.date].append(row.xs)
    days = list(grouped)
    if not days:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(days) for _ in days]
        values = [value for day in sample for value in grouped[day]]
        if values:
            estimates.append(sum(values) / len(values))
    if not estimates:
        return None
    estimates.sort()
    return [
        estimates[min(int(0.025 * len(estimates)), len(estimates) - 1)],
        estimates[min(int(0.975 * len(estimates)), len(estimates) - 1)],
    ]


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 3 or len(first) != len(second):
        return None
    first_ranks, second_ranks = _ranks(first), _ranks(second)
    return _correlation(first_ranks, second_ranks)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    numerator = sum(
        (a - mean_first) * (b - mean_second) for a, b in zip(first, second, strict=True)
    )
    first_sum = sum((a - mean_first) ** 2 for a in first)
    second_sum = sum((b - mean_second) ** 2 for b in second)
    denominator = sqrt(first_sum * second_sum)
    return numerator / denominator if denominator else None


def _t_value(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    deviation = sqrt(variance)
    if deviation == 0:
        return 0.0 if mean == 0 else None
    return mean / (deviation / sqrt(len(values)))


def _mean(values: Sequence[Any]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def _positive_horizons(value: Any) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("manifest horizons가 올바르지 않습니다.") from error
    if not result or any(item <= 0 for item in result):
        raise ValueError("horizons는 양수 목록이어야 합니다.")
    return tuple(dict.fromkeys(result))


def _mapping_value(value: Any, key: str, default: str) -> str:
    return str(value.get(key, default)) if isinstance(value, Mapping) else default


def _decimal_or_none(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _rate(value: Any) -> str:
    """0~1 사이 비율을 표시용 백분율로 변환한다."""

    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _value_percent(value: Any) -> str:
    """이미 퍼센트 단위인 가격 폭을 표시한다."""

    return "—" if value is None else f"{float(value):.1f}%"


def _percentage_points(value: Any) -> str:
    """초과 수익률을 퍼센트포인트로 표시한다."""

    return "—" if value is None else f"{float(value):.2f}%p"


def _rating_hit(metric: Mapping[str, Any]) -> str:
    return f"{metric.get('n', 0)} / {_rate(metric.get('hit_rate'))}"


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _interval(value: Any, interval: Any) -> str:
    if value is None:
        return "—"
    if not isinstance(interval, Sequence) or len(interval) != 2:
        return _percentage_points(value)
    return f"{float(value):.2f}%p [{float(interval[0]):.2f}, {float(interval[1]):.2f}]"
