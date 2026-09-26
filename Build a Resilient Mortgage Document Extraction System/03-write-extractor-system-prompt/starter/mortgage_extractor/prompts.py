"""System prompts, normalization rules, and few-shot examples.

The prompts in this module teach the extractor *how* to handle the cases the
schema alone cannot enforce: when to return null instead of fabricating, how to
normalize informal numerics, and how to record arithmetic discrepancies
verbatim rather than silently correcting them.

The single source of truth for normalization is :data:`NORMALIZATION_RULES`,
which is interpolated into every extractor system prompt and is the canonical
reference for the rules described in the module docstrings of
:mod:`mortgage_extractor.schema` and :mod:`mortgage_extractor.models`.
"""
from __future__ import annotations

from mortgage_extractor.models import DocumentType


NORMALIZATION_RULES = """
Normalization rules:
- Square footage: normalize informal square-footage expressions to numbers.
  Example: "about 2,400 sq ft" → 2400.
- Currency: remove currency symbols and thousands separators and return a
  numeric value. Example: "$485,000" → 485000.0.
- Percentage: convert percentages to decimal form. Example: "6.5%" → 0.065.
- Dates should be represented using the format expected by the schema.
- Numeric fields must contain numbers, not strings.
""".strip()


_CATEGORICAL_CRITERIA = """
1. Return null for any field not explicitly stated in the document. Do not infer, default, or fabricate.
2. Distinguish base income from bonus, commission, and overtime. Record each component only when the document explicitly identifies it.
3. When a categorical value does not fit an available enum, emit "other" and provide the corresponding *_detail value using the document's wording.
4. Numeric fields receive numbers, not strings. Normalize numeric expressions according to the normalization rules below.
""".strip()


_FEW_SHOT_EXAMPLES = """
<example name="clean">
<input>
Employee: Jane Smith
Base salary YTD: $72,000
Bonus YTD: $5,000
Commission YTD: $3,000
Overtime YTD: $1,200
</input>
<reasoning>
Every income component is explicitly stated, so each corresponding field can
be populated. Currency values are normalized to numeric values rather than
returned as strings.
</reasoning>
<output>
{
  "borrower": {
    "full_name": "Jane Smith",
    "coborrower_name": null
  },
  "income": {
    "base_ytd": 72000.0,
    "bonus_ytd": 5000.0,
    "total_ytd": 81200.0
  }
}
</output>
</example>

<example name="missing">
<input>
Employee: Robert Jones
Base salary YTD: $64,000
The employee has no bonus eligibility this year.
</input>
<reasoning>
bonus_ytd is null because the document does not report a bonus amount.
Fabricating zero would imply zero was earned or reported, rather than
preserving the distinction that the amount is absent from the document.
</reasoning>
<output>
{
  "borrower": {
    "full_name": "Robert Jones",
    "coborrower_name": null
  },
  "income": {
    "base_ytd": 64000.0,
    "bonus_ytd": null,
    "total_ytd": null
  }
}
</output>
</example>

<example name="informal">
<input>
Property appraisal:
The subject property contains about 2,400 sq ft of living area.
</input>
<reasoning>
The document explicitly states the approximate square footage. The word
"about" does not make the numeric value a string; the number should be
normalized to the integer 2400.
</reasoning>
<output>
{
  "property": {
    "address": null,
    "year_built": null,
    "property_type": null,
    "occupancy_type": null,
    "hoa_dues_monthly": null,
    "square_footage": 2400
  }
}
</output>
</example>

<example name="mismatch">
<input>
Employee: Maria Brown
Base salary YTD: $50,000
Bonus YTD: $4,000
Overtime YTD: $2,000
Total YTD: $60,000
</input>
<reasoning>
The stated total is preserved as stated rather than silently corrected. The
line items sum to $56,000, which differs from the document's stated $60,000.
The extractor must not replace the source value with its own arithmetic.
</reasoning>
<output>
{
  "borrower": {
    "full_name": "Maria Brown",
    "coborrower_name": null
  },
  "income": {
    "base_ytd": 50000.0,
    "bonus_ytd": 4000.0,
    "total_ytd": 60000.0
  }
}
</output>
</example>
""".strip()


def classifier_system_prompt() -> str:
    """System prompt for the classifier pass."""
    return (
        "You are a document classifier for a mortgage lender. You will receive "
        "the text of a single mortgage-related document and must call the "
        "`classify_document` tool exactly once with the document's type and a "
        "one-sentence reason describing the textual cues that drove the "
        "classification. Use `other` if the document is not one of the listed "
        "types or is too damaged to classify confidently."
    )


def extractor_system_prompt(doc_type: DocumentType) -> str:
    """Return the system prompt for the given document type's extractor."""
    if doc_type == DocumentType.INCOME_VERIFICATION:
        return income_verification_system_prompt()

    intro = (
        f"You are a mortgage document extractor handling a "
        f"{doc_type.value} document. Extract only information explicitly "
        f"stated in the document and call `extract_{doc_type.value}` exactly "
        f"once with the structured data. If the document is unreadable or "
        "cannot be reliably extracted, call `flag_for_review` instead."
    )

    return (
        intro
        + "\n\n"
        + NORMALIZATION_RULES
        + "\n\n"
        + _FEW_SHOT_EXAMPLES
    )


def income_verification_system_prompt() -> str:
    """Return the income-verification extractor system prompt."""
    intro = (
        "You are a mortgage income-verification document extractor. Extract "
        "only information explicitly stated in the document and call "
        "`extract_income_verification` exactly once with the structured data. "
        "If the document is unreadable or cannot be reliably extracted, call "
        "`flag_for_review` instead."
    )

    return (
        intro
        + "\n\n"
        + _CATEGORICAL_CRITERIA
        + "\n\n"
        + NORMALIZATION_RULES
        + "\n\n"
        + _FEW_SHOT_EXAMPLES
    )
