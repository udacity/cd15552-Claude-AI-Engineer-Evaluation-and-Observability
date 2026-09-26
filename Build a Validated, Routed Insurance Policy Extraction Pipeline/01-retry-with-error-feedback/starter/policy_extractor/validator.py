"""Validates extracted policy records and categorises failures.

The validator only handles semantic and consistency errors. JSON-syntax errors
are eliminated upstream by tool_choice-forced structured tool calls — the SDK
enforces the schema before this validator runs.
"""
from __future__ import annotations

from typing import Any

from policy_extractor.records import ValidationError


def validate_extraction(extraction: dict[str, Any]) -> ValidationError | None:
    """Return the first ValidationError found, or None if extraction is clean.

    Checks run in this order:
      1. Required fields absent (null) → missing_source
      2. Numeric fields out of legal range (negative premium, etc.) → format
      3. Cross-field consistency (premium vs sum of components) → consistency
    """
    error = _check_required_present(extraction)
    if error is not None:
        return error

    error = _check_numeric_ranges(extraction)
    if error is not None:
        return error

    return _check_premium_components_consistency(extraction)


_REQUIRED_FIELDS = ("premium_amount", "deductible", "coverage_limit", "endorsements", "exclusions")


def _check_required_present(extraction: dict[str, Any]) -> ValidationError | None:
    """Return missing_source ValidationError for the first null required field, else None."""
    for field in _REQUIRED_FIELDS:
        if extraction.get(field) is None:
            return ValidationError(
                field=field,
                category="missing_source",
                detected_pattern=f"{field}_absent",
                message=(
                    f"The source document does not contain {field}; "
                    "retry is futile because this information is missing from the source."
                ),
            )

    return None


def _check_numeric_ranges(extraction: dict[str, Any]) -> ValidationError | None:
    """Return format ValidationError for any negative numeric field, else None."""
    numeric_fields = (
        ("premium_amount", "negative_premium"),
        ("deductible", "negative_deductible"),
        ("coverage_limit", "negative_coverage_limit"),
    )

    for field, detected_pattern in numeric_fields:
        value = extraction.get(field)

        if isinstance(value, (int, float)) and value < 0:
            return ValidationError(
                field=field,
                category="format",
                detected_pattern=detected_pattern,
                message=f"{field} cannot be negative; extracted value was {value}.",
            )

    return None


def _check_premium_components_consistency(
    extraction: dict[str, Any],
) -> ValidationError | None:
    """Return consistency ValidationError when stated premium ≠ sum(components), else None."""
    components = extraction.get("premium_components")
    premium = extraction.get("premium_amount")

    if components and isinstance(premium, (int, float)):
        component_total = sum(
            component["amount"]
            for component in components
            if isinstance(component, dict)
            and isinstance(component.get("amount"), (int, float))
        )

        delta = component_total - premium

        if abs(delta) > 0.01:
            return ValidationError(
                field="premium_amount",
                category="consistency",
                detected_pattern="premium_does_not_match_components",
                message=(
                    f"Premium amount {premium} does not match the sum of "
                    f"premium components {component_total}."
                ),
            )

    return None
