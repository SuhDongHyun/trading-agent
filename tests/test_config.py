from pathlib import Path

from trading_agent.infrastructure.config import load_settings


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


def test_defaults_cover_api_models_and_ledger(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.yaml")

    assert settings.trading_api.base_url == "http://localhost:9999"
    assert settings.llm.claude.model == "claude-opus-5-5"
    assert settings.llm.codex.model == "gpt-6.1-sol"
    assert settings.ledger.db_path == Path("data/trading-agent.db")
