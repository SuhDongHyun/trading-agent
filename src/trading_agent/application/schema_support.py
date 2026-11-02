"""모델 JSON 응답을 엄격하게 읽는 공용 도구."""

import json
import math
from decimal import Decimal
from typing import Any, cast


def load_strict_json(raw_json: str) -> Any:
    """NaN·Infinity와 중복 키를 거절하고 실수는 Decimal로 읽는다."""

    return json.loads(
        raw_json,
        parse_float=Decimal,
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def require_object(value: object, required: set[str], label: str) -> dict[str, Any]:
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


def require_string(value: object, label: str) -> str:
    """문자열 필드를 확인한다."""

    if not isinstance(value, str):
        raise ValueError(f"{label}는 문자열이어야 합니다.")
    return value


def require_string_list(value: object, label: str) -> list[str]:
    """문자열 배열 필드를 확인한다."""

    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label}는 문자열 배열이어야 합니다.")
    return list(value)


def require_number(value: object, label: str) -> int | Decimal:
    """bool이 아닌 유한한 JSON 숫자를 확인한다."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{label}는 숫자여야 합니다.")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label}는 유한한 숫자여야 합니다.")
        return Decimal(str(value))
    return value


def require_optional_number(value: object, label: str) -> int | Decimal | None:
    """null 또는 유한한 JSON 숫자를 확인한다."""

    return None if value is None else require_number(value, label)


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
