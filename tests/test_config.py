from pathlib import Path

import pytest

from trading_agent.infrastructure.config import TradingApiSettings, load_settings


def test_yaml_settings_are_loaded(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "trading_api:\n  base_url: http://yaml.example\n"
        "watchlist:\n  - '005930'\n"
        "risk:\n  max_order_amount: 12345\n",
        encoding="utf-8",
    )

    settings = load_settings(config_file)

    assert settings.trading_api.base_url == "http://yaml.example"
    assert settings.watchlist == ["005930"]
    assert settings.risk.max_order_amount == 12345
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
