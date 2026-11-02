"""두 판단 CLI가 공유하는 응답 스키마와 엄격한 파서."""

from typing import Any

from trading_agent.application.schema_support import (
    load_strict_json,
    require_number,
    require_object,
    require_string,
    require_string_list,
)
from trading_agent.domain.models import ModelDecision, TradeProposal

DECISION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                    "quantity": {"type": "integer"},
                    "limit_price": {"type": "number"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": [
                    "code",
                    "action",
                    "quantity",
                    "limit_price",
                    "confidence",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "missing_info": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "proposals", "missing_info"],
    "additionalProperties": False,
}


def parse_model_decision(raw_json: str, model_id: str) -> ModelDecision:
    """JSON 응답의 타입과 필수 필드를 검증해 도메인 판단으로 만든다."""

    payload = require_object(
        load_strict_json(raw_json), {"summary", "proposals", "missing_info"}, "응답"
    )
    summary = require_string(payload["summary"], "summary")
    proposals = payload["proposals"]
    if not isinstance(proposals, list):
        raise ValueError("proposals는 배열이어야 합니다.")
    missing_info = require_string_list(payload["missing_info"], "missing_info")

    return ModelDecision(
        model_id=model_id,
        proposals=[
            _parse_proposal(item, index) for index, item in enumerate(proposals)
        ],
        summary=summary,
        missing_info=missing_info,
    )


_PROPOSAL_FIELDS = {"code", "action", "quantity", "limit_price", "confidence", "reason"}


def _parse_proposal(value: object, index: int) -> TradeProposal:
    """제안 object의 스키마와 도메인 불변식을 확인한다."""

    label = f"proposals[{index}]"
    proposal = require_object(value, _PROPOSAL_FIELDS, label)
    quantity = proposal["quantity"]
    if not isinstance(quantity, int) or isinstance(quantity, bool):
        raise ValueError(f"{label}.quantity는 정수여야 합니다.")
    return TradeProposal(
        code=require_string(proposal["code"], f"{label}.code"),
        action=require_string(proposal["action"], f"{label}.action"),  # type: ignore[arg-type]
        quantity=quantity,
        limit_price=require_number(proposal["limit_price"], f"{label}.limit_price"),
        confidence=require_number(proposal["confidence"], f"{label}.confidence"),  # type: ignore[arg-type]
        reason=require_string(proposal["reason"], f"{label}.reason"),
    )
