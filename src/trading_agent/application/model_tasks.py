"""두 CLI 모델에 맡기는 작업별 역할 지시, 응답 스키마, 파서."""

from trading_agent.application.decision_schema import (
    DECISION_RESPONSE_SCHEMA,
    parse_model_decision,
)
from trading_agent.application.ports import ModelTask
from trading_agent.application.prompt_builder import MODEL_SYSTEM_PROMPT
from trading_agent.application.recommendation_schema import (
    RECOMMENDATION_RESPONSE_SCHEMA,
    parse_model_recommendation,
)
from trading_agent.domain.models import ModelDecision
from trading_agent.domain.recommendation import ModelRecommendation

DECISION_TASK: ModelTask[ModelDecision] = ModelTask(
    name="decision",
    system_prompt=MODEL_SYSTEM_PROMPT,
    response_schema=DECISION_RESPONSE_SCHEMA,
    parse=parse_model_decision,
)

RECOMMENDATION_TASK: ModelTask[ModelRecommendation] = ModelTask(
    name="recommendation",
    system_prompt=MODEL_SYSTEM_PROMPT,
    response_schema=RECOMMENDATION_RESPONSE_SCHEMA,
    parse=parse_model_recommendation,
)
