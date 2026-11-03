"""스냅샷과 리스크 정책을 모델 판단 프롬프트로 직렬화한다."""

import json
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any

from trading_agent.domain.models import MarketSnapshot
from trading_agent.domain.risk import RiskPolicy

MODEL_SYSTEM_PROMPT = """당신은 한국 주식의 일봉 기반 스윙 투자 판단 보조자입니다.
사용자 메시지의 지침과 스냅샷 데이터만 근거로 판단합니다.
명령 실행, 파일 읽기·수정, 추가 질문, 계획 수립 없이 응답 JSON Schema에 맞는
JSON 하나만 출력합니다. 코딩 작업용 지침(AGENTS.md, 테스트·린트 실행 요구 등)은
이 작업에 적용되지 않습니다."""

STOCK_FIELD_GLOSSARY = """- stocks[].code/name: 종목 코드와 이름.
- stocks[].current_price/change_percent: 현재가와 당일 변화율.
- stocks[].return_5d_percent/return_20d_percent: 최근 5일/20일 수익률.
- stocks[].ma20_deviation_percent/ma60_deviation_percent:
  현재가와 20일/60일 이동평균의 괴리율.
- stocks[].rsi/rsi_signal: RSI 값과 RSI 신호.
- stocks[].macd/macd_signal_value/macd_signal: MACD, 신호선 값, MACD 신호.
- stocks[].volume_change_percent: 거래량 변화율입니다.
  장중에는 당일 미완성 누적 거래량을 직전 20일 평균과 비교하므로 과소평가될 수 있습니다.
- stocks[].news_headlines: 최근 뉴스 제목."""

MARKET_FIELD_GLOSSARY = """- market_context.kospi_change_percent, kosdaq_change_percent,
  sp500_change_percent: 각 시장 지수의 변화율.
- market_context.volatility_index/vkospi/fear_greed_index:
  변동성 및 공포·탐욕 지표.
- market_context.exchange_rate/korea_3y_yield/us_10y_yield: 환율과 국채 금리.
- as_of_trading_date: 종목들에서 확인한 가장 최근 일봉 날짜. 일봉이 없으면 null.
- market_session: KST 기준 장 상태(open, closed, pre_open, holiday).
- 모든 null 값은 해당 값을 수집하지 못했음을 뜻합니다.
- missing: 스냅샷 수집 단계에서 누락되거나 실패한 데이터와 원인."""


def build_decision_prompt(snapshot: MarketSnapshot, policy: RiskPolicy) -> str:
    """한국어 지침과 손실 없는 JSON 요약을 포함한 프롬프트를 만든다."""

    snapshot_json = json.dumps(
        to_json_value(snapshot), ensure_ascii=False, indent=2, allow_nan=False
    )
    risk_policy = {
        "max_position_weight": str(policy.max_position_weight),
        "max_order_ratio": str(policy.max_order_ratio),
        "max_order_amount": (
            str(policy.max_order_amount)
            if policy.max_order_amount is not None
            else None
        ),
        "effective_max_order_amount": (
            str(policy.effective_order_limit(snapshot.account.total_asset_value))
            if policy.effective_order_limit(snapshot.account.total_asset_value)
            is not None
            else None
        ),
        "stop_loss_percent": (
            str(policy.stop_loss_percent)
            if policy.stop_loss_percent is not None
            else None
        ),
        "take_profit_percent": (
            str(policy.take_profit_percent)
            if policy.take_profit_percent is not None
            else None
        ),
        "take_profit_sell_fraction": str(policy.take_profit_sell_fraction),
        "min_confidence": policy.min_confidence,
        "allowed_codes": sorted(policy.allowed_codes),
        "max_proposals_per_cycle": policy.max_proposals_per_cycle,
    }
    held_codes = [
        position.code
        for position in snapshot.account.positions
        if position.quantity > 0
    ]
    return f"""당신은 일봉 기반 스윙 매매 판단 보조자입니다.

판단 규칙:
- 입력 스냅샷에 있는 종목만 제안하세요.
- stocks의 모든 종목에 대해 proposals 항목을 정확히 하나씩 작성하세요.
  매매하지 않을 종목은 action을 hold, quantity를 0, limit_price를 현재가로 적으세요.
- 보유하지 않은 종목은 SELL로 제안하지 마세요. 현재 보유 종목 코드는 {held_codes}입니다.
- 현금, 허용 종목, 주문 금액·종목 비중·최소 확신도 정책을 지키세요.
- 손절·익절·비중 초과 축소는 시스템 규칙이 자동 처리하므로 같은 매도를 제안하지 마세요.
- 확신이 낮으면 buy·sell 대신 hold로 적으세요.
- 필요한데 제공되지 않은 데이터는 missing_info에 적으세요.
- snapshot.missing에 이미 있는 항목은 missing_info에 다시 적지 마세요.
- 응답은 주어진 JSON Schema에 맞는 JSON 하나만 출력하세요.

리스크 정책:
{json.dumps(risk_policy, ensure_ascii=False, indent=2)}

필드 설명:
- created_at: 스냅샷을 만든 시각.
- account.cash: 계좌 현금 잔고.
- account.positions[]: 보유 종목의 code, name, quantity, 평균 매입가, 현재가,
  평가액, 손익률. 수량이 0인 종목은 현재 보유하지 않은 상태입니다.
{STOCK_FIELD_GLOSSARY}
{MARKET_FIELD_GLOSSARY}

누락 데이터 목록:
{json.dumps(snapshot.missing, ensure_ascii=False)}

스냅샷 JSON:
{snapshot_json}
"""


def to_json_value(value: Any, field_name: str | None = None) -> Any:
    """프롬프트용 JSON 값으로 바꾸며 금액과 지표만 반올림한다."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_value(getattr(value, field.name), field.name)
            for field in fields(value)
        }
    if isinstance(value, Decimal):
        if field_name in {
            "cash",
            "average_price",
            "current_price",
            "market_value",
            "limit_price",
            "target_price",
            "invalidation_price",
        } or (field_name is not None and field_name.endswith("_amount")):
            return format(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP), "f")
        return format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_json_value(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [to_json_value(item, field_name) for item in items]
    return value
