import json
from decimal import Decimal

import pytest

from trading_agent.application.decision_schema import (
    DECISION_RESPONSE_SCHEMA,
    parse_model_decision,
)
from trading_agent.domain.errors import DomainError


def valid_response(**proposal_overrides: object) -> str:
    proposal = {
        "code": "005930",
        "action": "buy",
        "quantity": 3,
        "limit_price": 71000.25,
        "confidence": 0.8,
        "reason": "momentum",
    }
    proposal.update(proposal_overrides)
    return json.dumps(
        {"summary": "trend", "proposals": [proposal], "missing_info": ["news"]}
    )


def test_json_response_is_converted_to_domain_model() -> None:
    decision = parse_model_decision(valid_response(), "claude")

    assert decision.model_id == "claude"
    assert decision.proposals[0].quantity == 3
    assert decision.proposals[0].limit_price == Decimal("71000.25")
    assert decision.missing_info == ["news"]


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"summary":"x","proposals":[],"missing_info":[],"extra":true}',
        '{"summary":"x","proposals":[]}',
        "not JSON",
    ],
)
def test_parser_rejects_invalid_top_level_schema(raw_json: str) -> None:
    with pytest.raises(ValueError):
        parse_model_decision(raw_json, "codex")


@pytest.mark.parametrize(
    "proposal_overrides",
    [
        {"quantity": 1.5},
        {"quantity": "2"},
        {"quantity": True},
        {"limit_price": "100"},
        {"confidence": "0.8"},
        {"unknown": "value"},
    ],
)
def test_parser_rejects_wrong_types_and_extra_proposal_fields(
    proposal_overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        parse_model_decision(valid_response(**proposal_overrides), "claude")


@pytest.mark.parametrize(
    "proposal_overrides",
    [{"code": "ABCDEF"}, {"action": "purchase"}, {"quantity": 0}],
)
def test_parser_rejects_domain_invariant_violations(
    proposal_overrides: dict[str, object],
) -> None:
    with pytest.raises(DomainError):
        parse_model_decision(valid_response(**proposal_overrides), "claude")


def test_schema_forbids_extra_properties_and_requires_every_property() -> None:
    objects = [
        DECISION_RESPONSE_SCHEMA,
        DECISION_RESPONSE_SCHEMA["properties"]["proposals"]["items"],
    ]
    for schema in objects:
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
