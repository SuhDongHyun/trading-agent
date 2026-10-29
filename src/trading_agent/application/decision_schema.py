"""두 판단 CLI가 공유하는 응답 스키마와 엄격한 파서."""

import json
import math
from decimal import Decimal
from typing import Any, cast

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

    payload = _validate_object(
        json.loads(
            raw_json,
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        ),
        {"summary", "proposals", "missing_info"},
        "응답",
    )
    summary = payload["summary"]
    proposals = payload["proposals"]
    missing_info = payload["missing_info"]
    if not isinstance(summary, str):
        raise ValueError("summary는 문자열이어야 합니다.")
    if not isinstance(proposals, list):
        raise ValueError("proposals는 배열이어야 합니다.")
    if not isinstance(missing_info, list) or any(
        not isinstance(item, str) for item in missing_info
    ):
        raise ValueError("missing_info는 문자열 배열이어야 합니다.")

    parsed_proposals = [
        _parse_proposal(item, index) for index, item in enumerate(proposals)
    ]
    return ModelDecision(
        model_id=model_id,
        proposals=parsed_proposals,
        summary=summary,
        missing_info=missing_info,
    )


def _parse_proposal(value: object, index: int) -> TradeProposal:
    """제안 object의 스키마와 도메인 불변식을 확인한다."""

    required = {
        "code",
        "action",
        "quantity",
        "limit_price",
        "confidence",
        "reason",
    }
    proposal = _validate_object(value, required, f"proposals[{index}]")
    if not isinstance(proposal["code"], str):
        raise ValueError(f"proposals[{index}].code는 문자열이어야 합니다.")
    if not isinstance(proposal["action"], str):
        raise ValueError(f"proposals[{index}].action은 문자열이어야 합니다.")
    if not isinstance(proposal["quantity"], int) or isinstance(
        proposal["quantity"], bool
    ):
        raise ValueError(f"proposals[{index}].quantity는 정수여야 합니다.")
    limit_price = proposal["limit_price"]
    if isinstance(limit_price, bool) or not isinstance(
        limit_price, (int, float, Decimal)
    ):
        raise ValueError(f"proposals[{index}].limit_price는 숫자여야 합니다.")
    confidence = proposal["confidence"]
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float, Decimal)
    ):
        raise ValueError(f"proposals[{index}].confidence는 숫자여야 합니다.")
    if isinstance(confidence, float) and not math.isfinite(confidence):
        raise ValueError(f"proposals[{index}].confidence는 유한한 숫자여야 합니다.")
    if not isinstance(proposal["reason"], str):
        raise ValueError(f"proposals[{index}].reason은 문자열이어야 합니다.")

    return TradeProposal(
        code=proposal["code"],
        action=proposal["action"],  # type: ignore[arg-type]
        quantity=proposal["quantity"],
        limit_price=limit_price,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        reason=proposal["reason"],
    )


def _validate_object(value: object, required: set[str], label: str) -> dict[str, Any]:
    """object 필수 키만 허용해 스키마의 추가 필드 금지를 적용한다."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label}은 JSON object여야 합니다.")
    keys = set(value)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        details = []
        if missing:
            details.append(f"누락 필드: {', '.join(missing)}")
        if extra:
            details.append(f"알 수 없는 필드: {', '.join(extra)}")
        raise ValueError(f"{label} 스키마가 올바르지 않습니다 ({'; '.join(details)}).")
    return cast(dict[str, Any], value)


def _reject_constant(value: str) -> None:
    """표준 JSON이 아닌 NaN과 Infinity 리터럴을 거절한다."""

    raise ValueError(f"JSON 숫자 {value}는 허용되지 않습니다.")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """같은 object에서 중복된 JSON 키를 거절한다."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object에 중복 키 {key!r}가 있습니다.")
        result[key] = value
    return result
