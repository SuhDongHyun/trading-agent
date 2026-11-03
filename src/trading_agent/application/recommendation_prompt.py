"""계좌와 무관한 종목 추천 프롬프트를 만든다."""

import json
from collections.abc import Sequence

from trading_agent.application.prompt_builder import (
    CONFIDENCE_SCALE_GUIDANCE,
    DEFAULT_HORIZON_DAYS,
    MARKET_FIELD_GLOSSARY,
    PRICE_RECOMMENDATION_GUIDANCE,
    STOCK_FIELD_GLOSSARY,
    TRADE_DECISION_GUIDANCE,
    format_horizon_period,
    to_json_value,
)
from trading_agent.domain.models import ResearchSnapshot


def build_recommendation_prompt(
    snapshot: ResearchSnapshot,
    horizon_days: Sequence[int] = DEFAULT_HORIZON_DAYS,
) -> str:
    """후보 종목 전체에 대한 투자의견을 요청하는 프롬프트를 만든다."""

    snapshot_json = json.dumps(
        to_json_value(snapshot), ensure_ascii=False, indent=2, allow_nan=False
    )
    horizon_period = format_horizon_period(horizon_days)
    return f"""계좌 잔고나 보유 여부와 무관하게,
아래 후보 종목 각각의 투자의견을 내세요.
투자 기간은 일봉 기준 {horizon_period} 스윙입니다.

판단 규칙:
{TRADE_DECISION_GUIDANCE}
- stocks의 모든 종목에 대해 recommendations 항목을 정확히 하나씩 작성하세요.
  스냅샷에 없는 종목은 적지 마세요.
- rating은 buy, hold, sell 중 하나입니다.
  - buy: 지금 신규 진입해도 기대수익이 위험보다 크다.
  - hold: 진입·청산 근거가 약하거나 판단이 엇갈린다. 확신이 낮으면 hold로 적으세요.
  - sell: 보유 중이라면 줄이거나 피해야 한다.
- confidence는 아래 공통 확신도 척도에 따라 판단 자체의 확신도를 적으세요.
- buy와 sell이면 target_price와 invalidation_price를 원 단위 숫자로 적으세요.
  - buy: target_price는 목표가, invalidation_price는 손절가입니다.
    (손절가 < 현재가 < 목표가)
  - sell: target_price는 예상 하락 가격, invalidation_price는 그 위로 오르면
    매도 의견이 틀렸다고 볼 가격입니다. (예상 하락 가격 < 현재가 < 철회 가격)
  - hold는 둘 다 null로 적으세요.
- reason에는 근거가 된 지표와 위험 요인을 한두 문장으로 적으세요.
- 필요한데 제공되지 않은 데이터는 missing_info에 적되,
  아래 누락 데이터 목록에 이미 있는 항목은 다시 적지 마세요.
- 응답은 주어진 JSON Schema에 맞는 JSON 하나만 출력하세요.

{PRICE_RECOMMENDATION_GUIDANCE}

확신도 척도:
{CONFIDENCE_SCALE_GUIDANCE}

필드 설명:
- created_at: 스냅샷을 만든 시각.
{STOCK_FIELD_GLOSSARY}
{MARKET_FIELD_GLOSSARY}

누락 데이터 목록:
{json.dumps(snapshot.missing, ensure_ascii=False)}

스냅샷 JSON:
{snapshot_json}
"""
