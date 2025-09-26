#!/usr/bin/env python3
"""
Example 09: Refinement Basics
============================

This example demonstrates the basic refinement capabilities of LangStruct.
Refinement uses Best-of-N candidate selection and iterative improvement
to automatically boost extraction accuracy by 15-30%.

Key concepts:
- Simple refinement with refine=True
- Constructor vs method-level configuration
- Accuracy comparison (with/without refinement)
- Understanding refinement metadata
"""

import warnings
from typing import List, Optional

from pydantic import BaseModel, Field

from langstruct import LangStruct

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class InvoiceSchema(BaseModel):
    """Schema for invoice extraction"""

    invoice_number: str = Field(description="Complete invoice number with any prefixes")
    amount: float = Field(description="Invoice amount as decimal number")
    due_date: str = Field(description="Due date in YYYY-MM-DD format")
    vendor: str = Field(description="Company or person issuing the invoice")
    line_items: Optional[List[str]] = Field(description="List of products or services")


def main():
    print("🔧 LangStruct Refinement Basics")
    print("=" * 50)

    # Sample invoice text (intentionally tricky to extract perfectly)
    invoice_text = """
    INVOICE FROM: TechCorp Solutions Inc.

    Invoice #: INV-2024-789
    Amount Due: $3,450.00
    Payment Due: March 15th, 2024

    Services Provided:
    • Premium Software License
    • Installation & Setup Service
    • 12-month Technical Support Plan
    • Training Workshop (2 days)

    Please remit payment by the due date to avoid late fees.
    """

    print("📄 Sample Invoice Text:")
    print(invoice_text)
    print()

    # Create extractor
    extractor = LangStruct(schema=InvoiceSchema)

    # 1. Extraction WITHOUT refinement
    print("1️⃣  EXTRACTION WITHOUT REFINEMENT")
    print("-" * 40)

    result_basic = extractor.extract(invoice_text)

    print("📊 Basic Result:")
    for field, value in result_basic.entities.items():
        print(f"  {field}: {value}")

    print(f"\n💯 Confidence: {result_basic.confidence:.1%}")
    print(
        f"🔧 Refinement applied: {result_basic.metadata.get('refinement_applied', False)}"
    )
    print()

    # 2. Extraction WITH refinement (method-level)
    print("2️⃣  EXTRACTION WITH REFINEMENT")
    print("-" * 40)

    result_refined = extractor.extract(invoice_text, refine=True)

    print("📊 Refined Result:")
    for field, value in result_refined.entities.items():
        print(f"  {field}: {value}")

    print(f"\n💯 Confidence: {result_refined.confidence:.1%}")
    print(
        f"🔧 Refinement applied: {result_refined.metadata.get('refinement_applied', False)}"
    )

    # Show refinement metadata
    if result_refined.metadata.get("refinement_applied"):
        print(
            f"📈 Strategy: {result_refined.metadata.get('refinement_strategy', 'N/A')}"
        )
        print(
            f"🎯 Candidates generated: {result_refined.metadata.get('candidates_generated', 'N/A')}"
        )
        print(
            f"🔄 Refinement steps: {result_refined.metadata.get('refinement_steps', 'N/A')}"
        )
    print()

    # 3. Constructor-level refinement
    print("3️⃣  CONSTRUCTOR-LEVEL REFINEMENT")
    print("-" * 40)

    # Create extractor with refinement enabled by default
    refined_extractor = LangStruct(schema=InvoiceSchema, refine=True)

    result_constructor = refined_extractor.extract(invoice_text)

    print("📊 Constructor Refined Result:")
    for field, value in result_constructor.entities.items():
        print(f"  {field}: {value}")

    print(f"\n💯 Confidence: {result_constructor.confidence:.1%}")
    print(
        f"🔧 Refinement applied: {result_constructor.metadata.get('refinement_applied', False)}"
    )
    print()

    # 4. Compare results
    print("4️⃣  COMPARISON")
    print("-" * 40)

    print("📈 Accuracy Comparison:")

    # Count non-empty fields as a simple accuracy metric
    basic_fields = sum(1 for v in result_basic.entities.values() if v)
    refined_fields = sum(1 for v in result_refined.entities.values() if v)
    total_fields = len(result_basic.entities)

    print(f"  Basic extraction: {basic_fields}/{total_fields} fields filled")
    print(f"  Refined extraction: {refined_fields}/{total_fields} fields filled")

    confidence_improvement = result_refined.confidence - result_basic.confidence
    print(f"  Confidence improvement: {confidence_improvement:+.1%}")

    if refined_fields > basic_fields:
        print("✅ Refinement found more information!")
    elif result_refined.confidence > result_basic.confidence:
        print("✅ Refinement improved confidence!")
    else:
        print("ℹ️  Results similar - basic extraction was already good")

    print()

    # 5. Different text example
    print("5️⃣  CHALLENGING TEXT EXAMPLE")
    print("-" * 40)

    challenging_text = """
    From: Global Consulting LLC
    Bill dated 2024-02-28
    Ref: GC-2024-456

    Total: 2890 dollars
    Due by: end of March 2024

    Work completed:
    - Strategic planning consultation
    - Market analysis report
    - Competitive landscape study
    """

    print("📄 Challenging Text (less structured):")
    print(challenging_text)
    print()

    basic_challenge = extractor.extract(challenging_text)
    refined_challenge = extractor.extract(challenging_text, refine=True)

    print("📊 Basic vs Refined on Challenging Text:")
    print("\n🔹 Basic Result:")
    for field, value in basic_challenge.entities.items():
        print(f"  {field}: {value}")

    print("\n🔹 Refined Result:")
    for field, value in refined_challenge.entities.items():
        print(f"  {field}: {value}")

    # Calculate improvement
    basic_non_empty = sum(1 for v in basic_challenge.entities.values() if v)
    refined_non_empty = sum(1 for v in refined_challenge.entities.values() if v)

    if refined_non_empty > basic_non_empty:
        improvement = (
            (refined_non_empty - basic_non_empty) / len(basic_challenge.entities)
        ) * 100
        print(f"\n🎯 Refinement improved field completion by {improvement:.1f}%")

    print()
    print("✨ Key Takeaways:")
    print("  • Refinement typically improves accuracy by 15-30%")
    print("  • Works best on challenging or ambiguous text")
    print("  • Can be enabled at constructor or method level")
    print("  • Check refinement_applied metadata to confirm it ran")
    print("  • Higher confidence scores usually indicate better quality")


if __name__ == "__main__":
    main()
