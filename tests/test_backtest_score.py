from __future__ import annotations

import random
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from trading_agent.application.backtest import BACKTEST_FORMAT_VERSION
from trading_agent.application.backtest_score import (
    BacktestScoreOptions,
    BacktestScoreResult,
    ScoreBacktest,
    _calibration,
    _cluster_bootstrap,
    _date_ic,
    _LoadedDate,
    _Observation,
    _returns_by_horizon,
    _spearman,
    _t_value,
    market_index_for,
    price_outcome,
    render_score_markdown,
)
from trading_agent.application.ports import (
    DailyBar,
    IndexPoint,
    MarketIndex,
    StockQuote,
)
from trading_agent.domain.models import Action
from trading_agent.infrastructure.backtest_result_store import FileBacktestResultStore
from trading_agent.interface.bootstrap import BacktestScoreRuntime
from trading_agent.interface.cli import build_parser, main

DECISION_DATE = "20260630"
CODES = ("000001", "000002", "000003")


class MemoryScoreStore:
    def __init__(self, manifest: Mapping[str, Any], results: Mapping[str, Any]) -> None:
        self.manifest = manifest
        self.results = results

    def read_manifest(self) -> Mapping[str, Any]:
        return self.manifest

    def read_date_result(self, date: str) -> Mapping[str, Any] | None:
        return self.results.get(date)


class FakeScoreGateway:
    def __init__(self) -> None:
        self.quotes = {
            CODES[0]: StockQuote(CODES[0], "one", Decimal(100), Decimal(0), "KOSPI200"),
            CODES[1]: StockQuote(CODES[1], "two", Decimal(100), Decimal(0), "KSQ150"),
            CODES[2]: StockQuote(
                CODES[2], "three", Decimal(100), Decimal(0), "KOSDAQ150"
            ),
        }
        self.bars = {
            CODES[0]: [
                _bar("20260630", 100, 100, 100),
                _bar("20260701", 110, 120, 99),
                _bar("20260702", 105, 110, 90),
            ],
            CODES[1]: [
                _bar("20260630", 100, 100, 100),
                _bar("20260701", 100, 101, 99),
                _bar("20260702", 95, 96, 94),
            ],
            CODES[2]: [
                _bar("20260630", 100, 100, 100),
                _bar("20260701", 90, 110, 80),
                _bar("20260702", 95, 100, 88),
            ],
        }
        self.indexes = {
            MarketIndex.KOSPI: [
                IndexPoint("20260630", Decimal(1000)),
                IndexPoint("20260701", Decimal(1010)),
                IndexPoint("20260702", Decimal(1020)),
            ],
            MarketIndex.KOSDAQ: [
                IndexPoint("20260630", Decimal(1000)),
                IndexPoint("20260701", Decimal(980)),
                IndexPoint("20260702", Decimal(970)),
            ],
        }
        self.quote_calls: list[str] = []
        self.bar_calls: list[str] = []
        self.index_calls: list[MarketIndex] = []
        self.bar_ranges: list[tuple[str, str]] = []
        self.index_ranges: list[tuple[str, str]] = []

    def get_stock_quote(self, code: str) -> StockQuote:
        self.quote_calls.append(code)
        return self.quotes[code]

    def get_daily_bars(
        self, code: str, start_date: str, end_date: str
    ) -> list[DailyBar]:
        self.bar_calls.append(code)
        self.bar_ranges.append((start_date, end_date))
        return [bar for bar in self.bars[code] if start_date <= bar.date <= end_date]

    def get_market_index(
        self, index: MarketIndex, start_date: str, end_date: str
    ) -> list[IndexPoint]:
        self.index_calls.append(index)
        self.index_ranges.append((start_date, end_date))
        return [
            point
            for point in self.indexes[index]
            if start_date <= point.date <= end_date
        ]


def _bar(date: str, close: int, high: int, low: int) -> DailyBar:
    return DailyBar(
        date,
        Decimal(close),
        10,
        open_price=Decimal(close),
        high_price=Decimal(high),
        low_price=Decimal(low),
    )


def _recommendation(
    code: str, rating: str, confidence: float, target: int | None
) -> dict:
    invalidation = 95 if rating == "buy" else 105 if rating == "sell" else None
    return {
        "code": code,
        "rating": rating,
        "confidence": confidence,
        "target_price": str(target) if target is not None else None,
        "invalidation_price": str(invalidation) if invalidation is not None else None,
        "reason": "fixture",
        "consensus_basis": None,
    }


def _model(model_id: str, recommendations: list[dict]) -> dict:
    decision = {
        "model_id": model_id,
        "recommendations": recommendations,
        "summary": "fixture",
        "missing_info": [],
    }
    masked = {
        **decision,
        "recommendations": [
            {
                **item,
                "code": "900001",
                "target_price": (
                    str(Decimal(item["target_price"]) * 2)
                    if item["target_price"] is not None
                    else None
                ),
                "invalidation_price": (
                    str(Decimal(item["invalidation_price"]) * 2)
                    if item["invalidation_price"] is not None
                    else None
                ),
            }
            for item in recommendations
        ],
    }
    return {
        "model_id": model_id,
        "decision": masked,
        "restored_decision": decision,
    }


def _completed_result(date: str) -> dict:
    return {
        "format_version": BACKTEST_FORMAT_VERSION,
        "date": date,
        "status": "completed",
        "mode": "masked",
        "models": [
            _model(
                "claude",
                [
                    _recommendation(CODES[0], "buy", 0.6, 105),
                    _recommendation(CODES[1], "hold", 0.7, None),
                    _recommendation(CODES[2], "sell", 0.8, 80),
                ],
            ),
            _model(
                "codex",
                [
                    _recommendation(CODES[0], "hold", 0.7, None),
                    _recommendation(CODES[1], "hold", 0.8, None),
                    _recommendation(CODES[2], "sell", 0.9, 80),
                ],
            ),
        ],
        "leakage": {"daily_rows": 2, "news_articles": 1},
        "missing": [],
    }


def _score_fixture() -> tuple[MemoryScoreStore, FakeScoreGateway]:
    later = _completed_result("20260701")
    earlier = _completed_result(DECISION_DATE)
    failed = {
        "format_version": BACKTEST_FORMAT_VERSION,
        "date": "20260702",
        "status": "snapshot_only",
        "leakage": {"daily_rows": 1},
        "missing": ["news"],
    }
    manifest = {
        "format_version": BACKTEST_FORMAT_VERSION,
        "mode": "masked",
        "selected_dates": [DECISION_DATE, "20260701", "20260702"],
        "arguments": {"universe": list(CODES), "horizons": [1]},
        "consensus": {"policy": "buy_one_side", "confidence_aggregation": "min"},
    }
    return (
        MemoryScoreStore(
            manifest,
            {DECISION_DATE: earlier, "20260701": later, "20260702": failed},
        ),
        FakeScoreGateway(),
    )


def test_scoring_uses_future_prices_market_benchmark_and_mask_restored_models() -> None:
    store, gateway = _score_fixture()
    result = ScoreBacktest(
        store,
        gateway,
        BacktestScoreOptions(
            horizons=(1,),
            primary_horizons=(1,),
            compare_policies=True,
            bootstrap_iterations=100,
            hold_bands={1: 0.5},
        ),
        now=datetime.fromisoformat("2026-11-03T10:00:00+09:00"),
    ).run()

    assert result.data["mode"] == "masked"
    assert result.data["memory_leak_warning"] is True
    assert result.data["counts"]["completed_dates"] == 2
    assert result.data["counts"]["excluded_dates"] == 1
    assert result.data["counts"]["statuses"]["snapshot_only"] == 1
    assert result.data["counts"]["leakage_blocked_rows"] == 7
    assert result.data["counts"]["missing_benchmarks"] == 0
    assert "주요 horizon: 1거래일 (그 외는 참고)" in result.markdown
    assert result.data["periods"]["before_20260701"]["date_count"] == 1
    assert result.data["periods"]["from_20260701"]["date_count"] == 1
    assert set(gateway.quote_calls) == set(CODES)
    assert set(gateway.bar_calls) == set(CODES)
    assert set(gateway.index_calls) == {MarketIndex.KOSPI, MarketIndex.KOSDAQ}

    source = result.data["sources"]["consensus"]
    assert source["opinion_distribution"]["buy"] == 2
    assert source["opinion_distribution"]["sell"] == 2
    assert source["horizons"]["1"]["buy_n"] == 2
    assert source["horizons"]["1"]["by_rating"]["buy"]["n"] == 2
    assert source["horizons"]["1"]["by_rating"]["buy"]["hit_n"] == 1
    assert result.data["base_rates"]["1"]["xs_positive_rate"] == pytest.approx(1 / 3)
    assert result.data["base_rates"]["1"]["exc_positive_rate"] == pytest.approx(0.5)
    outcomes = result.data["price_outcomes"]["by_source"]
    assert outcomes["consensus"]["buy"]["target_first"] >= 1
    assert outcomes["consensus"]["buy"]["both_same_day"] >= 1
    assert outcomes["claude"]["buy"]["target_first"] >= 1
    assert outcomes["codex"]["sell"]["both_same_day"] >= 1
    assert (
        result.data["price_outcomes"]["width_medians_percent"]["consensus"]["buy"][
            "invalidation"
        ]
        < 0
    )
    assert (
        result.data["price_outcomes"]["width_medians_percent"]["consensus"]["sell"][
            "target"
        ]
        < 0
    )
    policies = {item["policy"]: item for item in result.data["policy_comparison"]}
    assert policies["strict"]["buy_n"] == 0
    assert policies["buy_one_side"]["buy_n"] == 2
    assert "## 의견 분포" in result.markdown
    assert "## 기준율" in result.markdown
    assert "## 목표가·무효화 가격 선착" in result.markdown
    assert "## 기간별 합의 성과" in result.markdown


def test_market_index_mapping_and_daily_return_exc_xs() -> None:
    assert market_index_for("KOSPI") is MarketIndex.KOSPI
    assert market_index_for("KOSPI200") is MarketIndex.KOSPI
    assert market_index_for("코스닥") is MarketIndex.KOSDAQ
    assert market_index_for("KSQ150") is MarketIndex.KOSDAQ
    assert market_index_for("KOSDAQ150") is MarketIndex.KOSDAQ
    assert market_index_for(None) is None
    dated = _LoadedDate("20260630", {}, {})
    prices = {
        (CODES[0], "all"): [
            _bar("20260630", 100, 100, 100),
            _bar("20260701", 110, 110, 110),
        ]
    }
    indexes = {
        (MarketIndex.KOSPI, "20260630"): Decimal(1000),
        (MarketIndex.KOSPI, "20260701"): Decimal(1010),
    }

    xs, exc, absolute = _returns_by_horizon(
        [dated], [CODES[0]], [1], prices, indexes, {CODES[0]: MarketIndex.KOSPI}
    )

    assert absolute[(dated.date, CODES[0], 1)] == pytest.approx(10)
    assert xs[(dated.date, CODES[0], 1)] == pytest.approx(0)
    assert exc[(dated.date, CODES[0], 1)] == pytest.approx(9)


def test_future_price_shortage_is_none_and_reported() -> None:
    dated = _LoadedDate("20260630", {}, {})
    prices = {
        (CODES[0], "all"): [
            _bar("20260630", 100, 100, 100),
            _bar("20260701", 110, 110, 110),
        ]
    }
    _, exc, absolute = _returns_by_horizon(
        [dated],
        [CODES[0]],
        [2],
        prices,
        {},
        {CODES[0]: MarketIndex.KOSPI},
    )
    assert absolute[(dated.date, CODES[0], 2)] is None
    assert exc[(dated.date, CODES[0], 2)] is None


def test_score_queries_never_cache_a_future_or_partial_date() -> None:
    store, gateway = _score_fixture()
    ScoreBacktest(
        store,
        gateway,
        BacktestScoreOptions(horizons=(1,), bootstrap_iterations=10),
        now=datetime.fromisoformat("2026-07-02T10:00:00+09:00"),
    ).run()

    assert gateway.bar_ranges
    assert gateway.index_ranges
    assert all(end == "20260701" for _, end in gateway.bar_ranges)
    assert all(end == "20260701" for _, end in gateway.index_ranges)


@pytest.mark.parametrize(
    ("action", "target", "stop", "bars", "expected"),
    [
        (Action.BUY, 110, 95, [_bar("20260701", 105, 112, 99)], "target_first"),
        (Action.BUY, 110, 95, [_bar("20260701", 94, 100, 90)], "invalidation_first"),
        (Action.BUY, 110, 95, [_bar("20260701", 100, 112, 90)], "both_same_day"),
        (Action.SELL, 90, 105, [_bar("20260701", 95, 100, 88)], "target_first"),
        (Action.SELL, 90, 105, [_bar("20260701", 106, 110, 95)], "invalidation_first"),
        (Action.SELL, 90, 105, [_bar("20260701", 100, 110, 88)], "both_same_day"),
        (Action.BUY, 110, 95, [_bar("20260701", 100, 105, 99)], "neither"),
        (Action.BUY, 110, 95, [DailyBar("20260701", Decimal(100), 10)], "no_price"),
    ],
)
def test_price_outcomes_for_both_directions(
    action: Action,
    target: int,
    stop: int,
    bars: list[DailyBar],
    expected: str,
) -> None:
    assert price_outcome(action, Decimal(target), Decimal(stop), bars) == expected


def test_price_outcome_requires_both_levels_and_complete_ohlc() -> None:
    good_bar = _bar("20260701", 100, 110, 90)
    assert price_outcome(Action.BUY, None, Decimal(95), [good_bar]) == "no_price"
    assert price_outcome(Action.BUY, Decimal(110), None, [good_bar]) == "no_price"
    missing_low = DailyBar("20260701", Decimal(100), 10, high_price=Decimal(110))
    assert (
        price_outcome(Action.BUY, Decimal(110), Decimal(95), [missing_low])
        == "no_price"
    )


def test_spearman_ties_date_ic_and_t_value() -> None:
    assert _spearman([1, 1, 2], [10, 10, 20]) == pytest.approx(1)
    assert _spearman([1, 1, 2], [20, 20, 10]) == pytest.approx(-1)
    assert _spearman([1, 2], [10, 20]) is None
    assert _t_value([0.1, 0.2, 0.3]) == pytest.approx(3.464101615)
    rows = [
        _Observation(
            "20260630", CODES[0], "claude", 5, Action.BUY, 0.9, 1, 0, None, None, None
        ),
        _Observation(
            "20260630", CODES[1], "claude", 5, Action.HOLD, 0.6, 0, 0, None, None, None
        ),
        _Observation(
            "20260630", CODES[2], "claude", 5, Action.SELL, 0.8, -1, 0, None, None, None
        ),
    ]
    assert _date_ic(rows)["mean"] == pytest.approx(1)


def test_bootstrap_is_seeded_and_interval_contains_mean() -> None:
    rows = [
        _Observation(
            "20260630", CODES[0], "claude", 5, Action.BUY, 0.8, -1, 0, None, None, None
        ),
        _Observation(
            "20260701", CODES[0], "claude", 5, Action.BUY, 0.8, 0, 0, None, None, None
        ),
        _Observation(
            "20260702", CODES[0], "claude", 5, Action.BUY, 0.8, 1, 0, None, None, None
        ),
    ]
    first = _cluster_bootstrap(rows, 500, 44)
    assert first == _cluster_bootstrap(rows, 500, 44)
    assert first is not None and first[0] <= 0 <= first[1]


def test_bootstrap_uses_harness_order_statistic_indices() -> None:
    rows = [
        _Observation(
            date,
            CODES[0],
            "claude",
            5,
            Action.BUY,
            0.8,
            value,
            0,
            None,
            None,
            None,
        )
        for date, value in zip(
            ("20260630", "20260701", "20260702"), (0, 0, 10), strict=True
        )
    ]
    iterations, seed = 31, 7
    grouped = {row.date: [row.xs] for row in rows}
    dates = list(grouped)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [
            value
            for day in (rng.choice(dates) for _ in dates)
            for value in grouped[day]
        ]
        means.append(sum(sample) / len(sample))
    means.sort()

    interval = _cluster_bootstrap(rows, iterations, seed)

    assert interval == [means[int(0.025 * len(means))], means[int(0.975 * len(means))]]


def test_confidence_bins_include_upper_edges_once() -> None:
    rows = [
        _Observation(
            "20260630",
            f"00000{index}",
            "claude",
            5,
            Action.BUY,
            confidence,
            1,
            0,
            None,
            None,
            None,
        )
        for index, confidence in enumerate((0.0, 0.59, 0.6, 0.7, 0.8, 1.0), start=1)
    ]
    buckets = _calibration(rows, 0.5)
    assert [bucket["bucket"] for bucket in buckets] == [
        "0.0-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-1.0",
    ]
    assert [bucket["n"] for bucket in buckets] == [2, 1, 1, 2]
    assert [bucket["hit_rate"] for bucket in buckets] == [1, 1, 1, 1]


def test_markdown_formats_rates_and_percentage_points_separately() -> None:
    markdown = render_score_markdown(
        {
            "mode": "plain",
            "policy": "strict",
            "confidence_aggregation": "min",
            "horizons": [5],
            "counts": {"completed_dates": 1, "excluded_dates": 0},
            "sources": {
                "consensus": {
                    "opinion_distribution": {"buy": 1, "hold": 1, "sell": 1},
                    "horizons": {
                        "5": {
                            "n": 3,
                            "direction_hit_rate": 0.6,
                            "buy_mean_xs": 7.8,
                            "buy_ci95": [2.0, 9.0],
                            "sell_mean_xs": -3.2,
                            "sell_ci95": [-4.0, -2.0],
                            "by_rating": {
                                "buy": {"n": 1, "hit_rate": 0.6},
                                "hold": {"n": 1, "hit_rate": 0.2},
                                "sell": {"n": 1, "hit_rate": 1.0},
                            },
                            "ic": {"mean": 0.3, "t_value": 1.2, "dates": 1},
                            "calibration": {
                                "buy_sell": [
                                    {
                                        "bucket": "0.6-0.7",
                                        "n": 3,
                                        "hit_rate": 0.6667,
                                        "exc_hit_rate": 0.5,
                                    }
                                ],
                                "hold": [],
                            },
                        }
                    },
                }
            },
            "base_rates": {
                "5": {
                    "universe_n": 10,
                    "xs_positive_rate": 0.432,
                    "xs_hold_band_rate": 0.2,
                    "exc_positive_rate": 0.7,
                    "absolute_positive_rate": 0.5,
                }
            },
            "price_outcomes": {
                "by_source": {
                    "consensus": {
                        "buy": {"target_first": 1},
                        "sell": {"both_same_day": 1},
                    }
                },
                "width_medians_percent": {
                    "consensus": {
                        "buy": {"invalidation": -5.5, "target": 7.8},
                        "sell": {"invalidation": 5.5, "target": -7.8},
                    }
                },
            },
            "disagreements": {
                "buy/hold": {
                    "n": 1,
                    "mean_xs": 7.8,
                    "claude_hit_rate": 0.6667,
                    "codex_hit_rate": 0.5,
                }
            },
            "policy_comparison": [
                {
                    "policy": "strict",
                    "confidence_aggregation": "min",
                    "buy_n": 1,
                    "buy_xs10": 7.8,
                    "buy_xs10_ci95": [2.0, 9.0],
                    "buy_xs20": 7.8,
                    "buy_xs20_ci95": [2.0, 9.0],
                    "sell_xs20": -3.2,
                }
            ],
            "periods": {
                "before_20260701": {"date_count": 1, "sources": {}},
                "from_20260701": {"date_count": 0, "sources": {}},
            },
        }
    )

    assert "60.0%" in markdown
    assert "43.2%" in markdown
    assert "66.7%" in markdown
    assert "7.80%p" in markdown
    assert "-5.5%" in markdown


def test_markdown_prioritizes_configured_investment_horizons() -> None:
    markdown = render_score_markdown(
        {
            "horizons": [5, 10, 20],
            "counts": {"completed_dates": 1, "excluded_dates": 0},
            "sources": {
                "consensus": {
                    "opinion_distribution": {},
                    "horizons": {str(horizon): {"n": 0} for horizon in (5, 10, 20)},
                }
            },
            "base_rates": {str(horizon): {} for horizon in (5, 10, 20)},
        },
        primary_horizons=(10, 20),
    )

    assert "주요 horizon: 10·20거래일 (그 외는 참고)" in markdown
    assert markdown.index("| consensus | 10 |") < markdown.index("| consensus | 20 |")
    assert markdown.index("| consensus | 20 |") < markdown.index("| consensus | 5 |")
    base_rates = markdown.split("## 기준율", maxsplit=1)[1].split("## IC", maxsplit=1)[
        0
    ]
    assert base_rates.index("| 10 |") < base_rates.index("| 20 |")
    assert base_rates.index("| 20 |") < base_rates.index("| 5 |")


def test_score_runtime_orders_primary_horizons_and_labels_references(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    horizons = (5, 10, 20)
    data = {
        "horizons": list(horizons),
        "policy": "buy_one_side",
        "confidence_aggregation": "mean",
        "counts": {"completed_dates": 1, "excluded_dates": 0},
        "sources": {
            "consensus": {
                "opinion_distribution": {},
                "horizons": {
                    str(horizon): {
                        "buy_n": 1,
                        "buy_mean_xs": horizon,
                        "sell_mean_xs": -horizon,
                    }
                    for horizon in horizons
                },
            }
        },
    }

    class FakeScoreUseCase:
        options = BacktestScoreOptions(primary_horizons=(10, 20))

        def run(self) -> BacktestScoreResult:
            return BacktestScoreResult(data, "rendered by use case")

    output_json = tmp_path / "score.json"
    runtime = BacktestScoreRuntime(
        FakeScoreUseCase(), object(), output_json, output_json.with_suffix(".md")
    )

    runtime.run()

    output = capsys.readouterr().out
    assert output.index("합의 10일:") < output.index("합의 20일:")
    assert output.index("합의 20일:") < output.index("합의 5일(참고):")
    assert output_json.with_suffix(".md").read_text(encoding="utf-8") == (
        "rendered by use case"
    )


def test_policy_comparison_includes_selected_policy_when_full_compare_is_off() -> None:
    store, gateway = _score_fixture()
    result = ScoreBacktest(
        store,
        gateway,
        BacktestScoreOptions(
            horizons=(1,),
            policy="strict",
            confidence_aggregation="mean",
            bootstrap_iterations=10,
        ),
        now=datetime.fromisoformat("2026-11-03T10:00:00+09:00"),
    ).run()
    assert result.data["policy_comparison"] == [
        {
            "policy": "strict",
            "confidence_aggregation": "mean",
            "buy_n": 0,
            "buy_xs10": None,
            "buy_xs10_ci95": None,
            "buy_xs20": None,
            "buy_xs20_ci95": None,
            "sell_xs20": None,
        }
    ]


def test_cli_arguments_output_files_and_offline_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parsed = build_parser().parse_args(
        [
            "backtest",
            "score",
            str(tmp_path),
            "--policy",
            "buy_one_side",
            "--confidence-aggregation",
            "mean",
            "--compare-policies",
            "--horizons",
            "5,10",
        ]
    )
    assert parsed.policy == "buy_one_side"
    assert parsed.confidence_aggregation == "mean"
    assert parsed.horizons == (5, 10)
    store, gateway = _score_fixture()
    output = tmp_path / "score.json"
    runtime = BacktestScoreRuntime(
        ScoreBacktest(
            store, gateway, BacktestScoreOptions(horizons=(1,), bootstrap_iterations=5)
        ),
        gateway,
        output,
        output.with_suffix(".md"),
    )
    runtime.run()
    assert output.is_file()
    assert output.with_suffix(".md").is_file()

    class FakeRuntime:
        called = False

        def run(self) -> None:
            self.called = True

        def close(self) -> None:
            pass

    instance = FakeRuntime()
    monkeypatch.setattr(
        "trading_agent.interface.cli.load_settings", lambda _path: object()
    )
    code = main(
        ["backtest", "score", str(tmp_path)],
        backtest_score_runtime_factory=lambda _settings, _args: instance,
    )
    assert code == 0
    assert instance.called


def test_markdown_and_file_store_round_trip(tmp_path: Path) -> None:
    store = FileBacktestResultStore(tmp_path)
    store.write_manifest(
        {"format_version": BACKTEST_FORMAT_VERSION, "selected_dates": []}
    )
    assert store.read_manifest() == {
        "format_version": BACKTEST_FORMAT_VERSION,
        "selected_dates": [],
    }
