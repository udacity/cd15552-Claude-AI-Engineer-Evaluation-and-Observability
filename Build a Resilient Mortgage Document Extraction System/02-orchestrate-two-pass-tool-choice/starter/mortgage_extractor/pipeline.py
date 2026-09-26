"""Two-pass extraction pipeline: classify, then extract.

The classification pass uses ``tool_choice={"type":"tool","name":...}`` — the
model is forced to call the classifier and cannot return free text. The
extraction pass uses ``tool_choice={"type":"any"}`` against a small set of
doc-type-specific tools (a primary extractor plus a ``flag_for_review`` escape
hatch). ``"any"`` rather than ``"auto"`` because ``"auto"`` would permit a
conversational text fallback; ``"any"`` guarantees a ``tool_use`` block.

This pattern mirrors the Architect's Playbook "broad-then-pinpoint" exploration
recipe: classify the document first, then drill into the doc-type-specific
extractor with a fallback for unrecoverable cases.
"""
from __future__ import annotations

import logging

from anthropic.types import Message, ToolUseBlock

from mortgage_extractor import prompts
from mortgage_extractor.client import RecordingClient
from mortgage_extractor.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from mortgage_extractor.errors import (
    ExtractionError,
    FlaggedForReviewError,
    UnsupportedDocumentTypeError,
)
from mortgage_extractor.models import (
    Classification,
    DocumentType,
    MortgageExtraction,
)
from mortgage_extractor.tools import (
    ToolDefinition,
    classify_document,
    doc_type_extractor,
    flag_for_review,
)

log = logging.getLogger(__name__)


class Pipeline:
    """Two-pass classifier + extractor over the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        client: RecordingClient | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.client = client or RecordingClient()
        self.max_tokens = max_tokens

    def run(self, document_text: str) -> MortgageExtraction:
    """Classify, then extract. Short-circuit on DocumentType.OTHER."""
    classification = self.classify_document(document_text)

    if classification.document_type == DocumentType.OTHER:
        raise UnsupportedDocumentTypeError(classification.reason)

    return self.extract(document_text, classification.document_type)

    def classify_document(self, document_text: str) -> Classification:
    """Pass 1: forced classifier call."""
    tool = classify_document()

    response = self.client.call(
        model=self.model,
        max_tokens=self.max_tokens,
        system=prompts.classifier_system_prompt(),
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": document_text}],
    )

    block = _single_tool_use_block(
        response,
        expected_name=tool["name"],
    )

    return Classification.model_validate(block.input)

    def extract(
    self,
    document_text: str,
    doc_type: DocumentType,
) -> MortgageExtraction:
    """Pass 2: tool_choice="any" extraction."""

    if doc_type == DocumentType.OTHER:
        raise UnsupportedDocumentTypeError(
            "extract() should not be called for DocumentType.OTHER"
        )

    extractor_tool = doc_type_extractor(doc_type)
    review_tool = flag_for_review()
    tools: list[ToolDefinition] = [extractor_tool, review_tool]

    response = self.client.call(
        model=self.model,
        max_tokens=self.max_tokens,
        system=prompts.extractor_system_prompt(doc_type),
        tools=tools,
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": document_text}],
    )

    block = _single_tool_use_block(response)

    if block.name == review_tool["name"]:
        reason = block.input.get("reason", "No reason provided")
        raise FlaggedForReviewError(reason)

    if block.name == extractor_tool["name"]:
        return MortgageExtraction.model_validate(block.input)

    raise ExtractionError(
        f"Unexpected tool called: {block.name!r}"
    )


def _single_tool_use_block(
    response: Message,
    *,
    expected_name: str | None = None,
) -> ToolUseBlock:
    """Pull the one ToolUseBlock out of an Anthropic Message response."""

    tool_blocks = [
        block
        for block in response.content
        if isinstance(block, ToolUseBlock)
    ]

    if not tool_blocks:
        actual_types = [type(block).__name__ for block in response.content]
        raise ExtractionError(
            f"Expected a ToolUseBlock but received content blocks: "
            f"{actual_types}"
        )

    if expected_name is None:
        return tool_blocks[0]

    for block in tool_blocks:
        if block.name == expected_name:
            return block

    actual_names = [block.name for block in tool_blocks]
    raise ExtractionError(
        f"Expected tool {expected_name!r}, but received tool calls: "
        f"{actual_names}"
    )
