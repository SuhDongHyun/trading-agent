"""스냅샷과 리스크 정책을 모델 판단 프롬프트로 직렬화한다."""

import json
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any

from trading_agent.domain.models import MarketSnapshot
from trading_agent.domain.risk import RiskPolicy

DEFAULT_HORIZON_DAYS = (10, 20)
OVERHEAT_RSI_THRESHOLD = 80
CHASE_BUY_RETURN_5D_THRESHOLD_PERCENT = 10

MODEL_SYSTEM_PROMPT = """당신은 한국 주식의 일봉 기반 스윙 투자 판단 보조자입니다.
사용자 메시지의 지침과 스냅샷 데이터만 근거로 판단합니다.
명령 실행, 파일 읽기·수정, 추가 질문, 계획 수립 없이 응답 JSON Schema에 맞는
JSON 하나만 출력합니다. 코딩 작업용 지침(AGENTS.md, 테스트·린트 실행 요구 등)은
이 작업에 적용되지 않습니다."""

STOCK_FIELD_GLOSSARY = """- stocks[].code/name: 종목 코드와 이름.
- stocks[].industry: 현재 종목 quote에 포함된 업종 메타데이터입니다.
- stocks[].current_price/change_percent: 현재가와 당일 변화율.
- stocks[].return_5d_percent/return_20d_percent: 최근 5일/20일 수익률.
- stocks[].relative_strength_5d_percent/relative_strength_20d_percent:
  종목 수익률에서 종목 소속 시장 지수(KOSPI 또는 KOSDAQ) 수익률을 뺀 초과수익(%p)입니다.
- stocks[].universe_rank_percentile_20d: 같은 스냅샷에서 20일 수익률이 유효한 종목의
  순위 백분위(0~100)입니다. 높을수록 강하며 동순위에는 평균 순위를 씁니다.
  비교 종목이 하나뿐이면 null입니다.
- stocks[].industry_relative_strength_20d_percent: 종목 20일 수익률에서 같은 업종의
  다른 종목 평균을 뺀 초과수익(%p)입니다. 비교할 동종 종목이 없으면 null입니다.
- stocks[].ma20_deviation_percent/ma60_deviation_percent:
  현재가와 20일/60일 이동평균의 괴리율.
- stocks[].rsi/rsi_signal: RSI 값과 RSI 신호.
- stocks[].macd/macd_signal_value/macd_signal: MACD, 신호선 값, MACD 신호.
- stocks[].volume_change_percent: 거래량 변화율입니다.
  장중에는 당일 미완성 누적 거래량을 직전 20일 평균과 비교하므로 과소평가될 수 있습니다.
- stocks[].atr14_percent: Wilder ATR(14)을 현재가로 나눈 일간 가격 변동 폭(%)입니다.
- stocks[].realized_volatility_20d_percent: 최근 20개 단순 일간 수익률의
  모집단 표준편차(%)입니다. 일간 기준이며 연율화하지 않았습니다.
- stocks[].high_20d_distance_percent: 현재가와 최근 20거래일 최고 고가의 거리(%)입니다.
  (현재가 − 최고 고가) ÷ 최고 고가이므로 최고가 아래면 음수입니다.
- stocks[].low_20d_distance_percent: 현재가와 최근 20거래일 최저 저가의 거리(%)입니다.
  (현재가 − 최저 저가) ÷ 최저 저가이므로 최저가 위면 양수입니다.
- stocks[].high_52w_position_percent: 최근 최대 252거래일 저가~고가 범위에서
  현재가가 차지하는 위치(%)입니다. 저가가 0%, 고가가 100%이며
  high_52w_period_trading_days가 실제 계산에 사용한 거래일 수입니다.
- market_session이 open이면 최신 일봉에 당일 미완성 봉이 포함될 수 있습니다.
- stocks[].news_headlines: 최근 뉴스 제목."""

PRICE_RECOMMENDATION_GUIDANCE = """가격 위험 규칙:
- ATR 금액은 현재가 × atr14_percent ÷ 100으로 계산하세요.
- 20일 최저 저가 금액은 현재가 ÷ (1 + low_20d_distance_percent ÷ 100),
  최고 고가 금액은 현재가 ÷ (1 + high_20d_distance_percent ÷ 100)으로 환산하세요.
- buy invalidation_price는 현재가 − 2×ATR 금액 이하 또는 최근 20거래일 최저 저가
  이하 중 더 낮은 가격을 기본으로 고려하세요.
- sell invalidation_price는 현재가 + 2×ATR 금액 이상 또는 최근 20거래일 최고 고가
  이상 중 더 높은 가격을 기본으로 고려하세요.
- 목표 폭 ÷ 손절 폭 손익비는 최소 1.5여야 합니다.
- ATR 또는 OHLC 근거가 없으면 임의로 계산하지 말고 missing_info에 적으세요."""

MARKET_FIELD_GLOSSARY = """- market_context.kospi_change_percent, kosdaq_change_percent,
  sp500_change_percent: 각 시장 지수의 변화율.
- market_context.kospi_return_5d_percent/kospi_return_20d_percent와
  kosdaq_return_5d_percent/kosdaq_return_20d_percent: 각 지수의 최근 5/20거래일
  수익률(%)입니다.
- market_context.kospi_ma20_deviation_percent/kospi_ma60_deviation_percent와
  kosdaq_ma20_deviation_percent/kosdaq_ma60_deviation_percent: 최신 지수 값과
  최근 20/60거래일 단순 이동평균의 괴리율(%)입니다. 양수면 이동평균 위입니다.
- market_context.vkospi_change_5d_points: 현재 VKOSPI에서 5거래일 전 값을 뺀
  포인트 변화입니다.
- 지수가 60일선 위에 있고 20일 수익률이 양수면 추세장, 반대면 약세·반전장으로 보고
  같은 종목 지표도 시장 국면에 따라 다르게 해석하세요.
- market_context.volatility_index/vkospi/fear_greed_index:
  변동성 및 공포·탐욕 지표.
- market_context.exchange_rate/korea_3y_yield/us_10y_yield: 환율과 국채 금리.
- as_of_trading_date: 종목들에서 확인한 가장 최근 일봉 날짜. 일봉이 없으면 null.
- market_session: KST 기준 장 상태(open, closed, pre_open, holiday).
- 모든 null 값은 해당 값을 수집하지 못했음을 뜻합니다.
- missing: 스냅샷 수집 단계에서 누락되거나 실패한 데이터와 원인."""

CONFIDENCE_SCALE_GUIDANCE = """- confidence는 0~1 사이의 판단 자체에 대한 확신도입니다.
  0.5는 근거가 엇갈려 동전 던지기 수준, 0.6은 약한 우위,
  0.7은 같은 근거로 10번 중 7번 맞을 수준, 0.8은 여러 근거가 같은 방향,
  0.9 이상은 거의 확실한 경우로 드물게 사용하세요.
  hold의 confidence도 매수·매도하지 않기로 한 판단 자체의 확신도입니다."""

TRADE_DECISION_GUIDANCE = f"""- stocks[].rsi가 {OVERHEAT_RSI_THRESHOLD} 이상이거나
  stocks[].ma20_deviation_percent 또는 stocks[].ma60_deviation_percent의 이격이
  크다는 이유만으로 sell하지 마세요. 강한 추세에서는 과열이 오래 지속될 수 있습니다.
- sell은 다음 추세 훼손 신호 중 하나 이상이 함께 확인될 때만 내세요:
  stocks[].macd가 stocks[].macd_signal_value를 하향 교차해 stocks[].macd_signal이
  sell로 표시되는 경우, stocks[].ma20_deviation_percent가 0 미만인 종가의 20일선
  이탈, 또는 stocks[].change_percent가 0 미만이면서
  stocks[].volume_change_percent가 0 초과인 거래량 증가 동반 하락.
- stocks[].return_5d_percent가 {CHASE_BUY_RETURN_5D_THRESHOLD_PERCENT}% 이상인
  급등 직후에는 추격 매수 위험이 큽니다. 눌림 후 지지 확인이나 거래량 동반 돌파처럼
  별도의 매수 근거가 하나 더 있을 때만 buy하세요."""


def format_horizon_period(horizon_days: Sequence[int] = DEFAULT_HORIZON_DAYS) -> str:
    """거래일 horizon 목록을 사람이 읽는 투자 기간으로 바꾼다."""

    values = tuple(horizon_days)
    if (
        not values
        or any(type(value) is not int or value <= 0 for value in values)
        or values != tuple(sorted(values))
        or len(values) != len(set(values))
    ):
        raise ValueError(
            "horizon_days는 오름차순의 중복 없는 양의 정수 목록이어야 합니다."
        )
    if len(values) == 1:
        return f"{values[0]}거래일"

    def weeks(days: int) -> str:
        return format(Decimal(days) / Decimal(5), "f").rstrip("0").rstrip(".")

    return (
        f"{values[0]}~{values[-1]}거래일(약 {weeks(values[0])}~{weeks(values[-1])}주)"
    )


def build_decision_prompt(
    snapshot: MarketSnapshot,
    policy: RiskPolicy,
    horizon_days: Sequence[int] = DEFAULT_HORIZON_DAYS,
) -> str:
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
    horizon_period = format_horizon_period(horizon_days)
    return f"""당신은 일봉 기반 스윙 매매 판단 보조자입니다.
투자 기간은 일봉 기준 {horizon_period} 스윙입니다.

판단 규칙:
- 입력 스냅샷에 있는 종목만 제안하세요.
- stocks의 모든 종목에 대해 proposals 항목을 정확히 하나씩 작성하세요.
  매매하지 않을 종목은 action을 hold, quantity를 0, limit_price를 현재가로 적으세요.
- 보유하지 않은 종목은 SELL로 제안하지 마세요. 현재 보유 종목 코드는 {held_codes}입니다.
- 현금, 허용 종목, 주문 금액·종목 비중·최소 확신도 정책을 지키세요.
- 손절·익절·비중 초과 축소는 시스템 규칙이 자동 처리하므로 같은 매도를 제안하지 마세요.
- 과열·급등 및 추세 훼손 판단에는 아래 공통 매매 원칙을 먼저 적용하세요.
- 매매 근거가 약하거나 판단 확신이 낮으면 buy·sell 대신 hold로 적으세요.
- 필요한데 제공되지 않은 데이터는 missing_info에 적으세요.
- snapshot.missing에 이미 있는 항목은 missing_info에 다시 적지 마세요.
- 각 제안의 confidence는 아래 공통 척도를 따르세요.
- 응답은 주어진 JSON Schema에 맞는 JSON 하나만 출력하세요.

확신도 척도:
{CONFIDENCE_SCALE_GUIDANCE}

공통 매매 원칙:
{TRADE_DECISION_GUIDANCE}

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
            "original_invalidation_price",
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
