"""Anthropic tool definitions for the mortgage extraction pipeline.

A "tool" in the Anthropic Messages API is a JSON object with a ``name``,
``description``, and ``input_schema`` (JSON Schema). When the API returns a
``tool_use`` content block, its ``input`` is guaranteed to validate against
this schema, which is how this project enforces structured output.

The canonical extractor tool, :func:`extract_mortgage_data`, will be registered
for the second pass of the pipeline in Exercise 2. The classifier tool,
:func:`classify_document`, will be registered for the forced first pass. Both
share the schema you build in :mod:`mortgage_extractor.schema`.
"""
from __future__ import annotations

from typing import TypedDict

from mortgage_extractor.models import DocumentType
from mortgage_extractor.schema import JsonSchema, mortgage_data_schema


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: JsonSchema


def extract_mortgage_data() -> ToolDefinition:
    """Return the canonical mortgage-data extractor tool definition."""
    return {
        "name": "extract_mortgage_data",
        "description": (
            "Extract the mortgage data stated in the document. "
            "Return null for any field that the document does not state; "
            "do not fabricate or infer missing values. For categorical "
            "fields, use 'other' when none of the enum values fit and "
            "provide the corresponding *_detail value with the document's "
            "free-text category."
        ),
        "input_schema": mortgage_data_schema(),
    }


def classify_document() -> ToolDefinition:
    """Return the document-classifier tool definition."""
    return {
        "name": "classify_document",
        "description": (
            "Classify the document as loan_application, appraisal, "
            "income_verification, or other. Provide a brief reason for "
            "the classification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "enum": [
                        "loan_application",
                        "appraisal",
                        "income_verification",
                        "other",
                    ],
                },
                "reason": {
                    "type": "string",
                },
            },
            "required": [
                "document_type",
                "reason",
            ],
        },
    }


def doc_type_extractor(doc_type: DocumentType) -> ToolDefinition:
    """Return a doc-type-tailored extractor tool."""
    schema = mortgage_data_schema()
    schema["required"] = _required_sections_for(doc_type)

    return {
        "name": f"extract_{doc_type.value}",
        "description": (
            f"Extract mortgage data from this {doc_type.value} document. "
            "Return null for fields the document does not state; never "
            "fabricate missing information. For categorical fields, use "
            "'other' when no enum value fits and provide the corresponding "
            "*_detail value."
        ),
        "input_schema": schema,
    }


def flag_for_review() -> ToolDefinition:
    """Return the escape-hatch tool the model calls when it cannot extract."""
    return {
        "name": "flag_for_review",
        "description": (
            "Flag the document for human review when the required mortgage "
            "data cannot be reliably extracted. Provide the reason."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                },
            },
            "required": ["reason"],
        },
    }


def _required_sections_for(doc_type: DocumentType) -> list[str]:
    """Which top-level sections must appear in the extractor's output."""
    if doc_type == DocumentType.LOAN_APPLICATION:
        return ["borrower", "property", "loan"]

    if doc_type == DocumentType.APPRAISAL:
        return ["property"]

    if doc_type == DocumentType.INCOME_VERIFICATION:
        return ["borrower", "income"]

    if doc_type == DocumentType.OTHER:
        raise ValueError(
            "OTHER documents do not have a document-type extractor"
        )

    raise ValueError(f"Unsupported document type: {doc_type}")


__all__ = [
    "ToolDefinition",
    "classify_document",
    "doc_type_extractor",
    "extract_mortgage_data",
    "flag_for_review",
]
