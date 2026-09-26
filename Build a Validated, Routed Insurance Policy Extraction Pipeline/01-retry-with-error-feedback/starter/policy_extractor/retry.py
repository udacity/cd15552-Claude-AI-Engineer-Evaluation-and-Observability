"""Retry-with-error-feedback orchestration for policy extraction."""
from __future__ import annotations

import logging
from typing import Any

from policy_extractor.client import MessageClient
from policy_extractor.extractor import (
    EXTRACT_POLICY_TOOL,
    build_extraction_messages,
    parse_tool_use,
)
from policy_extractor.records import (
    Endorsement,
    ExtractionOutcome,
    PolicyExtraction,
    PremiumComponent,
    RetryFutileEscalation,
    ValidationError,
)
from policy_extractor.validator import validate_extraction

logger = logging.getLogger(__name__)


DEFAULT_EXTRACTOR_MODEL = "claude-haiku-4-5-20251001"


def extract_with_retry(
    *,
    client: MessageClient,
    policy_id: str,
    document_text: str,
    max_retries: int = 3,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_tokens: int = 2048,
) -> ExtractionOutcome:
    """Extract one policy with validation-driven retry.

    Returns PolicyExtraction on success, RetryFutileEscalation when the source
    document is missing required information (no retry attempted).
    """
    prior_attempts: list[dict[str, Any]] = []
    history: list[ValidationError] = []
    last_error: ValidationError | None = None

    for attempt_index in range(max_retries + 1):
        messages, system = build_extraction_messages(
            document_text,
            prior_attempts,
        )

        response = client.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=[EXTRACT_POLICY_TOOL],
            tool_choice={
                "type": "tool",
                "name": "extract_policy",
            },
        )

        extraction = parse_tool_use(response)
        error = validate_extraction(extraction)

        if error is None:
            return build_extraction(
                policy_id=policy_id,
                extraction=extraction,
                attempt_index=attempt_index,
                history=history,
            )

        if error.category == "missing_source":
            return RetryFutileEscalation(
                policy_id=policy_id,
                reason=error.message,
                detected_pattern=error.detected_pattern,
            )

        history.append(error)
        last_error = error

        prior_attempts.append(
            {
                "extraction": extraction,
                "error_field": error.field,
                "error_category": error.category,
                "error_pattern": error.detected_pattern,
                "error_message": error.message,
            }
        )

    assert last_error is not None

    return RetryFutileEscalation(
        policy_id=policy_id,
        reason=last_error.message,
        detected_pattern=f"retries_exhausted__{last_error.detected_pattern}",
    )

def build_extraction(
    *,
    policy_id: str,
    extraction: dict[str, Any],
    attempt_index: int,
    history: list[ValidationError],
) -> PolicyExtraction:
    """Marshal a validated extraction dict into a PolicyExtraction dataclass record."""
    raw_endorsements = extraction.get("endorsements")
    if raw_endorsements is None:
        endorsements: list[Endorsement] | None = None
    else:
        endorsements = [
            Endorsement(name=e["name"], limit=e.get("limit")) for e in raw_endorsements
        ]

    raw_components = extraction.get("premium_components")
    components: list[PremiumComponent] | None = None
    if raw_components is not None:
        components = [
            PremiumComponent(name=c["name"], amount=c["amount"]) for c in raw_components
        ]

    return PolicyExtraction(
        policy_id=policy_id,
        policy_type=extraction["policy_type"],
        premium_amount=extraction["premium_amount"],
        deductible=extraction["deductible"],
        coverage_limit=extraction["coverage_limit"],
        endorsements=endorsements,
        exclusions=list(extraction.get("exclusions") or []),
        premium_components=components,
        confidence=dict(extraction.get("confidence") or {}),
        retry_count=attempt_index,
        final_attempt_index=attempt_index,
        validation_history=list(history),
    )
