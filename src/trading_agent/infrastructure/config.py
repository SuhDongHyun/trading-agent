"""YAML과 환경변수로부터 애플리케이션 설정을 읽는다."""

import math
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, Field, StrictInt, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from trading_agent.domain.models import ConfidenceAggregation, ConsensusPolicyName

_DEFAULT_CONFIG_FILE = Path("config.yaml")
_MARKET_TIMEOUT_KEYS = {
    "kospi",
    "kosdaq",
    "sp500",
    "vix",
    "vkospi",
    "fear_and_greed",
    "usd_krw",
    "korea_3y_yield",
    "us_10y_yield",
}


class YamlSettingsSource(PydanticBaseSettingsSource):
    """지정한 YAML 파일의 최상위 설정을 읽는다."""

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_file: Path,
    ) -> None:
        super().__init__(settings_cls)
        self.yaml_file = yaml_file

    def get_field_value(
        self,
        field: Any,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        """필드별 조회 대신 파일 전체를 반환한다."""

        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """YAML 설정을 dict로 반환한다."""

        if not self.yaml_file.is_file():
            return {}
        with self.yaml_file.open(encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{self.yaml_file}의 최상위 값은 mapping이어야 합니다.")
        return data


class TradingApiSettings(BaseModel):
    """읽기 전용 trading-api 접속 설정."""

    base_url: str = "http://localhost:9999"
    timeout_seconds: float = Field(default=10, gt=0)
    market_timeouts: dict[str, float] = Field(default_factory=dict)

    @field_validator("market_timeouts")
    @classmethod
    def validate_market_timeouts(cls, values: dict[str, float]) -> dict[str, float]:
        """지원 지표만 양수 timeout으로 설정했는지 확인한다."""

        unknown = values.keys() - _MARKET_TIMEOUT_KEYS
        if unknown:
            raise ValueError(
                "지원하지 않는 market_timeouts 지표: " + ", ".join(sorted(unknown))
            )
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("market_timeouts 값은 0보다 커야 합니다.")
        return values

    def timeout_for(self, indicator: str) -> float:
        """지표별 재정의 값이 없으면 기본 API timeout을 반환한다."""

        return self.market_timeouts.get(indicator, self.timeout_seconds)


class RiskSettings(BaseModel):
    """사이클에 적용할 리스크 한도."""

    max_position_weight: float = Field(default=0.2, gt=0, le=1)
    max_order_ratio: float = Field(default=0.1, gt=0, le=1)
    max_order_amount: float | None = Field(default=None, gt=0)
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    max_proposals_per_cycle: int = Field(default=5, gt=0)
    stop_loss_percent: float | None = Field(default=None, gt=0)
    take_profit_percent: float | None = Field(default=None, gt=0)
    take_profit_sell_fraction: float = Field(default=0.5, gt=0, le=1)


class ClaudeSettings(BaseModel):
    """Claude CLI 모델 설정."""

    command: str = "claude"
    model: str = "claude-opus-5-5"
    effort: str = "high"
    timeout_seconds: float = Field(default=300, gt=0)


class CodexSettings(BaseModel):
    """Codex CLI 모델 설정."""

    command: str = "codex"
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "high"
    timeout_seconds: float = Field(default=300, gt=0)


class LLMSettings(BaseModel):
    """두 판단 모델의 설정."""

    claude: ClaudeSettings = Field(default_factory=ClaudeSettings)
    codex: CodexSettings = Field(default_factory=CodexSettings)


class RecommendationSettings(BaseModel):
    """계좌와 무관한 종목 추천 설정."""

    universe: list[str] = Field(default_factory=list)
    max_results: int = Field(default=5, gt=0)
    top_candidates: int = Field(default=5, gt=0)
    horizon_days: list[StrictInt] = Field(default_factory=lambda: [10, 20])

    @field_validator("horizon_days")
    @classmethod
    def validate_horizon_days(cls, values: list[int]) -> list[int]:
        """투자 기간 horizon을 오름차순의 양의 정수로 제한한다."""

        if (
            not values
            or any(value <= 0 for value in values)
            or values != sorted(values)
            or len(values) != len(set(values))
        ):
            raise ValueError(
                "horizon_days는 오름차순의 중복 없는 양의 정수 목록이어야 합니다."
            )
        return values


class SnapshotSettings(BaseModel):
    """스냅샷 뉴스 수집 범위와 프롬프트 항목 수."""

    news_lookback_trading_days: int = Field(default=3, ge=1, le=10)
    news_limit: int = Field(default=5, ge=1, le=20)

    @field_validator("news_lookback_trading_days", "news_limit", mode="before")
    @classmethod
    def reject_non_integer_news_settings(cls, value: Any) -> Any:
        """boolean과 실수는 거부하면서 정수형 환경변수 문자열은 허용한다."""

        if isinstance(value, (bool, float)):
            raise ValueError("뉴스 설정은 정수여야 합니다.")
        return value


class ConsensusSettings(BaseModel):
    """합의 정책과 확신도 집계 설정."""

    policy: ConsensusPolicyName = ConsensusPolicyName.BUY_ONE_SIDE
    confidence_aggregation: ConfidenceAggregation = ConfidenceAggregation.MEAN


class PriceGuardSettings(BaseModel):
    """추천 손절 폭과 목표/손절 손익비 정책."""

    mode: Literal["adjust", "warn", "off"] = "adjust"
    min_stop_atr_multiple: float = Field(default=2.0, gt=0)
    min_reward_risk: float = Field(default=1.5, gt=0)


class LedgerSettings(BaseModel):
    """결정 기록 저장 위치."""

    db_path: Path = Path("data/trading-agent.db")
    stale_run_after_seconds: int = Field(default=3600, gt=0)


class PaperSettings(BaseModel):
    """가상 계좌 모의투자 설정."""

    initial_cash: float = Field(default=10_000_000, gt=0)
    commission_rate: float = Field(default=0.00015, ge=0, lt=1)
    sell_tax_rate: float = Field(default=0.002, ge=0, lt=1)
    slippage_rate: float = Field(default=0.0, ge=0, lt=1)
    db_path: Path = Path("data/paper/portfolio.db")
    ledger_db_path: Path = Path("data/paper/decisions.db")


class AgentSettings(BaseSettings):
    """환경변수가 YAML 설정을 덮어쓰는 애플리케이션 설정."""

    _yaml_file: ClassVar[Path] = _DEFAULT_CONFIG_FILE

    trading_api: TradingApiSettings = Field(default_factory=TradingApiSettings)
    watchlist: list[str] = Field(default_factory=list)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    recommendation: RecommendationSettings = Field(
        default_factory=RecommendationSettings
    )
    snapshot: SnapshotSettings = Field(default_factory=SnapshotSettings)
    consensus: ConsensusSettings = Field(default_factory=ConsensusSettings)
    price_guard: PriceGuardSettings = Field(default_factory=PriceGuardSettings)
    ledger: LedgerSettings = Field(default_factory=LedgerSettings)
    paper: PaperSettings = Field(default_factory=PaperSettings)

    model_config = SettingsConfigDict(
        env_prefix="TRADING_AGENT__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def recommendation_universe(self) -> list[str]:
        """추천 후보가 비어 있으면 watchlist를 후보로 쓴다."""

        return list(self.recommendation.universe or self.watchlist)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """환경변수와 YAML을 기본 우선순위로 합친다."""

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls, cls._yaml_file),
            file_secret_settings,
        )


def load_settings(config_file: str | Path | None = None) -> AgentSettings:
    """기본 또는 지정한 YAML 파일과 환경변수를 병합한다."""

    if config_file is None:
        return AgentSettings()

    selected_file = Path(config_file).expanduser().resolve()
    if not selected_file.is_file():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {selected_file}")

    class ConfiguredAgentSettings(AgentSettings):
        _yaml_file = selected_file

    return ConfiguredAgentSettings()
