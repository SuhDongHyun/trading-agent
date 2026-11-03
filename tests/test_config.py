from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_agent.domain.models import ConfidenceAggregation, ConsensusPolicyName
from trading_agent.infrastructure.config import (
    SnapshotSettings,
    TradingApiSettings,
    load_settings,
)


def test_yaml_settings_are_loaded(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "trading_api:\n  base_url: http://yaml.example\n"
        "watchlist:\n  - '005930'\n"
        "risk:\n  max_order_amount: 12345\n  min_confidence: 0.75\n",
        encoding="utf-8",
    )

    settings = load_settings(config_file)

    assert settings.trading_api.base_url == "http://yaml.example"
    assert settings.watchlist == ["005930"]
    assert settings.risk.max_order_amount == 12345
    assert settings.risk.min_confidence == 0.75
    assert settings.risk.max_order_ratio == 0.1


def test_environment_overrides_yaml_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "trading_api:\n  base_url: http://yaml.example\n"
        "risk:\n  max_order_amount: 12345\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_AGENT__TRADING_API__BASE_URL", "http://env.example")
    monkeypatch.setenv("TRADING_AGENT__RISK__MAX_ORDER_AMOUNT", "54321")

    settings = load_settings(config_file)

    assert settings.trading_api.base_url == "http://env.example"
    assert settings.risk.max_order_amount == 54321


def test_default_config_is_loaded_from_current_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        "watchlist:\n  - '005930'\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    settings = load_settings()

    assert settings.watchlist == ["005930"]
    assert settings.trading_api.base_url == "http://localhost:9999"
    assert settings.llm.claude.model == "claude-opus-5-5"
    assert settings.llm.codex.model == "gpt-6.1-sol"
    assert settings.llm.claude.timeout_seconds == 300
    assert settings.llm.codex.timeout_seconds == 300
    assert settings.ledger.db_path == Path("data/trading-agent.db")
    assert settings.ledger.stale_run_after_seconds == 3600
    assert settings.consensus.policy is ConsensusPolicyName.BUY_ONE_SIDE
    assert settings.consensus.confidence_aggregation is ConfidenceAggregation.MEAN
    assert settings.risk.min_confidence == 0.6
    assert settings.price_guard.mode == "adjust"
    assert settings.price_guard.min_stop_atr_multiple == 2.0
    assert settings.price_guard.min_reward_risk == 1.5


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="설정 파일을 찾을 수 없습니다"):
        load_settings(tmp_path / "missing.yaml")


def test_missing_default_config_uses_builtin_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.watchlist == []
    assert settings.risk.max_order_amount is None
    assert settings.risk.max_order_ratio == 0.1
    assert settings.risk.stop_loss_percent is None
    assert settings.risk.take_profit_percent is None
    assert settings.risk.take_profit_sell_fraction == 0.5


def test_recommendation_horizon_defaults_and_environment_override(
    tmp_path: Path, monkeypatch
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")

    settings = load_settings(config_file)
    assert settings.recommendation.horizon_days == [10, 20]
    assert settings.recommendation.top_candidates == 5
    assert settings.snapshot.news_lookback_trading_days == 3
    assert settings.snapshot.news_limit == 5

    monkeypatch.setenv("TRADING_AGENT__RECOMMENDATION__HORIZON_DAYS", "[5, 15]")
    settings = load_settings(config_file)
    assert settings.recommendation.horizon_days == [5, 15]


@pytest.mark.parametrize(
    "horizon_days",
    [[], [0], [-1], [20, 10], [10, 10], [True], [10.5]],
)
def test_recommendation_horizon_rejects_invalid_values(
    horizon_days: list[object],
) -> None:
    from trading_agent.infrastructure.config import RecommendationSettings

    with pytest.raises(ValidationError):
        RecommendationSettings(horizon_days=horizon_days)


@pytest.mark.parametrize("value", [0, -1])
def test_recommendation_top_candidates_must_be_positive(value: int) -> None:
    from trading_agent.infrastructure.config import RecommendationSettings

    with pytest.raises(ValidationError):
        RecommendationSettings(top_candidates=value)


def test_snapshot_news_settings_yaml_and_environment_override(
    tmp_path: Path, monkeypatch
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "snapshot:\n  news_lookback_trading_days: 4\n  news_limit: 8\n",
        encoding="utf-8",
    )

    settings = load_settings(config_file)
    assert settings.snapshot.news_lookback_trading_days == 4
    assert settings.snapshot.news_limit == 8

    monkeypatch.setenv("TRADING_AGENT__SNAPSHOT__NEWS_LOOKBACK_TRADING_DAYS", "2")
    monkeypatch.setenv("TRADING_AGENT__SNAPSHOT__NEWS_LIMIT", "12")
    settings = load_settings(config_file)
    assert settings.snapshot.news_lookback_trading_days == 2
    assert settings.snapshot.news_limit == 12


@pytest.mark.parametrize("value", [0, -1, 11, True, 5.5])
def test_snapshot_news_lookback_must_be_one_to_ten_integer(value: object) -> None:
    with pytest.raises(ValidationError):
        SnapshotSettings(news_lookback_trading_days=value)


@pytest.mark.parametrize("value", [0, 21, True, 5.5])
def test_snapshot_news_limit_must_be_from_one_through_twenty(value: object) -> None:
    with pytest.raises(ValidationError):
        SnapshotSettings(news_limit=value)


def test_market_indicator_timeout_override_uses_default_when_missing() -> None:
    settings = TradingApiSettings(
        timeout_seconds=10,
        market_timeouts={"vkospi": 2.5, "vix": 4},
    )

    assert settings.timeout_for("vkospi") == 2.5
    assert settings.timeout_for("vix") == 4
    assert settings.timeout_for("kospi") == 10


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_market_indicator_timeout_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="market_timeouts"):
        TradingApiSettings(market_timeouts={"vkospi": value})


def test_market_indicator_timeout_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="지원하지 않는"):
        TradingApiSettings(market_timeouts={"unknown": 2})


def test_consensus_environment_overrides_yaml(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "consensus:\n  policy: strict\n  confidence_aggregation: min\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_AGENT__CONSENSUS__POLICY", "buy_one_side")
    monkeypatch.setenv("TRADING_AGENT__CONSENSUS__CONFIDENCE_AGGREGATION", "mean")

    settings = load_settings(config_file)

    assert settings.consensus.policy is ConsensusPolicyName.BUY_ONE_SIDE
    assert settings.consensus.confidence_aggregation is ConfidenceAggregation.MEAN


@pytest.mark.parametrize(
    ("key", "value"),
    [("policy", "majority"), ("confidence_aggregation", "max")],
)
def test_consensus_settings_reject_invalid_values(
    tmp_path: Path, key: str, value: str
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"consensus:\n  {key}: {value}\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(config_file)


def test_price_guard_yaml_settings_are_loaded(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "price_guard:\n  mode: warn\n"
        "  min_stop_atr_multiple: 2.5\n  min_reward_risk: 1.8\n",
        encoding="utf-8",
    )

    settings = load_settings(config_file)

    assert settings.price_guard.mode == "warn"
    assert settings.price_guard.min_stop_atr_multiple == 2.5
    assert settings.price_guard.min_reward_risk == 1.8


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "modify"},
        {"min_stop_atr_multiple": 0},
        {"min_reward_risk": 0},
    ],
)
def test_price_guard_settings_reject_invalid_values(values: dict[str, object]) -> None:
    from trading_agent.infrastructure.config import PriceGuardSettings

    with pytest.raises(ValidationError):
        PriceGuardSettings(**values)
