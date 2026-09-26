"""JSON Schema for mortgage document extraction.

The schema is a single canonical structure spanning all three document types
(loan application, appraisal, income verification). Per-document-type extractor
tools share these properties but differ in their ``required`` lists.

Nullable fields use the union-type idiom (``type: ["<base>", "null"]``) so the
model can return ``null`` for absent fields rather than fabricating values.
Categorical fields that may grow over time use the ``enum + "other" + *_detail``
pattern: when none of the enum members fit, the model emits ``"other"`` and
writes a free-text reason into the sibling ``_detail`` field.
"""
from __future__ import annotations

from typing import Any

JsonSchema = dict[str, Any]


PROPERTY_TYPES: list[str] = [
    "single_family",
    "condo",
    "townhouse",
    "multi_family",
    "manufactured",
    "other",
]

OCCUPANCY_TYPES: list[str] = [
    "primary_residence",
    "second_home",
    "investment",
    "other",
]

LOAN_PURPOSES: list[str] = [
    "purchase",
    "refinance_rate_term",
    "refinance_cash_out",
    "other",
]


def mortgage_data_schema() -> JsonSchema:
    """Return the canonical JSON Schema for mortgage data extraction."""
    return {
        "type": "object",
        "properties": {
            "borrower": {
                "type": "object",
                "properties": {
                    "full_name": {
                        "type": "string",
                    },
                    "coborrower_name": {
                        "type": ["string", "null"],
                    },
                },
                "required": ["full_name"],
            },
            "property": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                    },
                    "year_built": {
                        "type": ["integer", "null"],
                    },
                    "property_type": {
                        "type": "string",
                        "enum": PROPERTY_TYPES,
                    },
                    "property_type_detail": {
                        "type": ["string", "null"],
                    },
                    "occupancy_type": {
                        "type": "string",
                        "enum": OCCUPANCY_TYPES,
                    },
                    "occupancy_type_detail": {
                        "type": ["string", "null"],
                    },
                    "hoa_dues_monthly": {
                        "type": ["number", "null"],
                    },
                },
                "required": ["address"],
            },
            "loan": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                    },
                    "purpose": {
                        "type": "string",
                        "enum": LOAN_PURPOSES,
                    },
                    "purpose_detail": {
                        "type": ["string", "null"],
                    },
                },
                "required": ["amount"],
            },
            "income": {
                "type": "object",
                "properties": {
                    "base_ytd": {
                        "type": ["number", "null"],
                    },
                    "bonus_ytd": {
                        "type": ["number", "null"],
                    },
                    "total_ytd": {
                        "type": ["number", "null"],
                    },
                },
                "required": [],
            },
        },
        "required": ["borrower", "property", "loan", "income"],
    }


def list_nullable_fields(schema: JsonSchema) -> list[str]:
    """Return dotted paths of every nullable leaf field in the schema."""
    result: list[str] = []

    def walk(node: JsonSchema, prefix: str = "") -> None:
        properties = node.get("properties", {})

        for name, field in properties.items():
            path = f"{prefix}.{name}" if prefix else name
            field_type = field.get("type")

            if isinstance(field_type, list) and "null" in field_type:
                result.append(path)

            elif field_type == "object":
                walk(field, path)

    walk(schema)
    return result
