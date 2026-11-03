"""종목 추천 응답 스키마와 엄격한 파서."""

from typing import Any

from trading_agent.application.schema_support import (
    load_strict_json,
    require_number,
    require_object,
    require_optional_number,
    require_string,
    require_string_list,
)
from trading_agent.domain.recommendation import (
    ModelRecommendation,
    StockRecommendation,
)

_NULLABLE_NUMBER = {"type": ["number", "null"]}
_ITEM_FIELDS = {
    "code",
    "rating",
    "confidence",
    "target_price",
    "invalidation_price",
    "reason",
}

RECOMMENDATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "rating": {"type": "string", "enum": ["buy", "sell", "hold"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "target_price": _NULLABLE_NUMBER,
                    "invalidation_price": _NULLABLE_NUMBER,
                    "reason": {"type": "string"},
                },
                "required": sorted(_ITEM_FIELDS),
                "additionalProperties": False,
            },
        },
        "missing_info": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "recommendations", "missing_info"],
    "additionalProperties": False,
}


def parse_model_recommendation(raw_json: str, model_id: str) -> ModelRecommendation:
    """JSON 응답을 검증해 도메인 투자의견 목록으로 만든다."""

    payload = require_object(
        load_strict_json(raw_json),
        {"summary", "recommendations", "missing_info"},
        "응답",
    )
    items = payload["recommendations"]
    if not isinstance(items, list):
        raise ValueError("recommendations는 배열이어야 합니다.")
    return ModelRecommendation(
        model_id=model_id,
        recommendations=[_parse_item(item, index) for index, item in enumerate(items)],
        summary=require_string(payload["summary"], "summary"),
        missing_info=require_string_list(payload["missing_info"], "missing_info"),
    )


def _parse_item(value: object, index: int) -> StockRecommendation:
    """추천 항목의 스키마와 도메인 불변식을 확인한다."""

    label = f"recommendations[{index}]"
    item = require_object(value, _ITEM_FIELDS, label)
    return StockRecommendation(
        code=require_string(item["code"], f"{label}.code"),
        rating=require_string(item["rating"], f"{label}.rating"),  # type: ignore[arg-type]
        confidence=require_number(item["confidence"], f"{label}.confidence"),  # type: ignore[arg-type]
        target_price=require_optional_number(
            item["target_price"], f"{label}.target_price"
        ),  # type: ignore[arg-type]
        invalidation_price=require_optional_number(
            item["invalidation_price"], f"{label}.invalidation_price"
        ),  # type: ignore[arg-type]
        reason=require_string(item["reason"], f"{label}.reason"),
    )
