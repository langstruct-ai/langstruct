#!/usr/bin/env python3
"""
Example 11: Refinement Comparison Study
=======================================

This example demonstrates the effectiveness of refinement across different
document types and complexity levels. See real accuracy improvements
and understand when refinement provides the most value.

Compares extraction quality across:
- Simple, well-structured documents
- Complex, poorly-formatted text
- Documents with missing information
- Different domains (financial, legal, medical, technical)
"""

import json
import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langstruct import LangStruct

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ContractSchema(BaseModel):
    """Legal contract information"""

    party_1: str = Field(description="First party name")
    party_2: str = Field(description="Second party name")
    contract_type: str = Field(description="Type of contract")
    effective_date: str = Field(description="When contract becomes effective")
    term_length: str = Field(description="Duration of the contract")
    termination_clause: bool = Field(description="Whether early termination is allowed")


class TechnicalSchema(BaseModel):
    """Technical documentation"""

    product_name: str = Field(description="Name of the product or system")
    version: str = Field(description="Version number")
    requirements: List[str] = Field(description="System requirements")
    installation_steps: int = Field(description="Number of installation steps")
    supported_platforms: List[str] = Field(description="Supported operating systems")


class FinancialSchema(BaseModel):
    """Financial document information"""

    company_name: str = Field(description="Company name")
    reporting_period: str = Field(description="Quarter or year being reported")
    revenue: float = Field(description="Total revenue in millions")
    profit_margin: float = Field(description="Profit margin as percentage")
    key_metrics: List[str] = Field(description="Important financial metrics mentioned")


def calculate_completeness(entities: Dict[str, Any]) -> float:
    """Calculate what percentage of fields are filled"""
    filled = sum(1 for v in entities.values() if v is not None and v != [] and v != "")
    total = len(entities)
    return (filled / total) * 100 if total > 0 else 0


def calculate_accuracy_score(entities: Dict[str, Any]) -> float:
    """Rough accuracy score based on field types and completeness"""
    score = 0.0
    total_weight = 0.0

    for field, value in entities.items():
        weight = 1.0
        field_score = 0.0

        if value is None or value == "" or value == []:
            field_score = 0.0  # Empty field
        elif isinstance(value, str):
            if len(value.split()) >= 2:  # Multi-word strings are usually better
                field_score = 1.0
            else:
                field_score = 0.7
        elif isinstance(value, (int, float)) and value > 0:
            field_score = 1.0  # Valid numbers
        elif isinstance(value, list) and len(value) > 0:
            field_score = 1.0  # Non-empty lists
        elif isinstance(value, bool):
            field_score = 1.0  # Booleans are binary, so valid if present
        else:
            field_score = 0.5  # Something is there but unclear quality

        score += field_score * weight
        total_weight += weight

    return (score / total_weight) * 100 if total_weight > 0 else 0


def compare_extraction(extractor: LangStruct, text: str, document_type: str) -> Dict:
    """Compare basic vs refined extraction and return metrics"""

    print(f"📄 {document_type}")
    print("-" * 60)
    print(f"Text preview: {text[:150]}..." if len(text) > 150 else text)
    print()

    # Basic extraction
    basic_result = extractor.extract(text)
    basic_completeness = calculate_completeness(basic_result.entities)
    basic_accuracy = calculate_accuracy_score(basic_result.entities)

    # Refined extraction
    refined_result = extractor.extract(text, refine=True)
    refined_completeness = calculate_completeness(refined_result.entities)
    refined_accuracy = calculate_accuracy_score(refined_result.entities)

    # Show results
    print("🔹 Basic Extraction:")
    for field, value in basic_result.entities.items():
        print(f"  {field}: {value}")
    print(f"  💯 Confidence: {basic_result.confidence:.1%}")
    print(f"  📊 Completeness: {basic_completeness:.1f}%")
    print()

    print("🔹 Refined Extraction:")
    for field, value in refined_result.entities.items():
        print(f"  {field}: {value}")
    print(f"  💯 Confidence: {refined_result.confidence:.1%}")
    print(f"  📊 Completeness: {refined_completeness:.1f}%")

    # Show refinement metadata
    if refined_result.metadata.get("refinement_applied"):
        print(f"  ⚙️ Strategy: {refined_result.metadata.get('refinement_strategy')}")
        print(f"  🎯 Candidates: {refined_result.metadata.get('candidates_generated')}")

    print()

    # Calculate improvements
    completeness_improvement = refined_completeness - basic_completeness
    confidence_improvement = refined_result.confidence - basic_result.confidence
    accuracy_improvement = refined_accuracy - basic_accuracy

    improvement_summary = {
        "document_type": document_type,
        "basic_completeness": basic_completeness,
        "refined_completeness": refined_completeness,
        "completeness_improvement": completeness_improvement,
        "basic_confidence": basic_result.confidence * 100,
        "refined_confidence": refined_result.confidence * 100,
        "confidence_improvement": confidence_improvement * 100,
        "accuracy_improvement": accuracy_improvement,
        "basic_entities": basic_result.entities,
        "refined_entities": refined_result.entities,
    }

    print("📈 Improvement Summary:")
    print(
        f"  🎯 Completeness: {completeness_improvement:+.1f}% ({basic_completeness:.1f}% → {refined_completeness:.1f}%)"
    )
    print(
        f"  💯 Confidence: {confidence_improvement:+.1%} ({basic_result.confidence:.1%} → {refined_result.confidence:.1%})"
    )
    print(f"  📊 Quality Score: {accuracy_improvement:+.1f}%")

    if completeness_improvement > 5:
        print("  ✅ Significant completeness improvement!")
    elif confidence_improvement > 0.05:
        print("  ✅ Notable confidence boost!")
    elif completeness_improvement > 0:
        print("  ➕ Minor improvement")
    else:
        print("  ➖ Limited improvement (basic extraction was already good)")

    print("\n" + "=" * 80 + "\n")

    return improvement_summary


def main():
    print("📊 Refinement Comparison Study")
    print("=" * 50)
    print("Comparing extraction accuracy across different document types\n")

    # Test cases with different complexity levels
    test_cases = [
        {
            "schema": ContractSchema,
            "type": "Legal Contract (Well-Structured)",
            "text": """
            SERVICE AGREEMENT

            This Service Agreement ("Agreement") is entered into on January 15, 2024,
            between TechCorp Solutions LLC ("Company") and Global Services Inc ("Client").

            Term: This agreement shall remain in effect for a period of 24 months
            from the effective date.

            Termination: Either party may terminate this agreement with 30 days
            written notice.
            """,
        },
        {
            "schema": ContractSchema,
            "type": "Legal Contract (Poorly Formatted)",
            "text": """
            contract stuff between alice johnson consulting and
            big retail company effective sometime in march 2024

            gonna last for about 1 year maybe longer depending on how things go

            either side can bail out if they want but should probably give some notice
            this is a consulting agreement for marketing services
            """,
        },
        {
            "schema": TechnicalSchema,
            "type": "Technical Documentation (Complete)",
            "text": """
            CloudSync Pro v2.1.4

            System Requirements:
            - Windows 10 or later
            - macOS 11.0 or later
            - Linux (Ubuntu 20.04+)
            - 4GB RAM minimum
            - 10GB free disk space

            Installation Process (5 steps):
            1. Download installer
            2. Run setup wizard
            3. Enter license key
            4. Configure sync settings
            5. Complete initial backup

            Supported on Windows, Mac, and Linux platforms.
            """,
        },
        {
            "schema": TechnicalSchema,
            "type": "Technical Documentation (Incomplete)",
            "text": """
            New software release...

            You'll need some recent computer, probably works on most systems.
            Setup is pretty straightforward, just run the thing.

            Should work on whatever you're using.
            """,
        },
        {
            "schema": FinancialSchema,
            "type": "Financial Report (Detailed)",
            "text": """
            Apple Inc. Q3 2024 Earnings Report

            Revenue: $125.3 billion for the third quarter
            Profit margin: 23.4%

            Key highlights:
            - iPhone sales exceeded expectations
            - Services revenue grew 15% year-over-year
            - Operating cash flow reached $32.1 billion
            - Strong performance in international markets
            """,
        },
        {
            "schema": FinancialSchema,
            "type": "Financial Report (Ambiguous)",
            "text": """
            Some company did pretty well last quarter, made lots of money.

            Sales were up compared to last time, margins looked good.
            The important numbers were better than expected.
            Cash flow was solid, international business growing.
            """,
        },
    ]

    # Run comparison for each test case
    results = []

    for test_case in test_cases:
        extractor = LangStruct(schema=test_case["schema"])
        result = compare_extraction(extractor, test_case["text"], test_case["type"])
        results.append(result)

    # Overall analysis
    print("📊 OVERALL ANALYSIS")
    print("=" * 50)

    # Calculate averages
    avg_completeness_improvement = sum(
        r["completeness_improvement"] for r in results
    ) / len(results)
    avg_confidence_improvement = sum(
        r["confidence_improvement"] for r in results
    ) / len(results)

    print(f"📈 Average Improvements:")
    print(f"  🎯 Completeness: {avg_completeness_improvement:+.1f}%")
    print(f"  💯 Confidence: {avg_confidence_improvement:+.1f}%")
    print()

    # Find best and worst cases
    best_case = max(results, key=lambda r: r["completeness_improvement"])
    worst_case = min(results, key=lambda r: r["completeness_improvement"])

    print("🏆 Best Improvement:")
    print(f"  Document: {best_case['document_type']}")
    print(f"  Completeness gain: {best_case['completeness_improvement']:+.1f}%")
    print()

    print("🤔 Least Improvement:")
    print(f"  Document: {worst_case['document_type']}")
    print(f"  Completeness gain: {worst_case['completeness_improvement']:+.1f}%")
    print()

    # Insights
    print("💡 Key Insights:")

    # When refinement helps most
    poorly_formatted = [
        r
        for r in results
        if "Poorly" in r["document_type"]
        or "Incomplete" in r["document_type"]
        or "Ambiguous" in r["document_type"]
    ]
    well_formatted = [r for r in results if r not in poorly_formatted]

    if poorly_formatted:
        avg_poor = sum(r["completeness_improvement"] for r in poorly_formatted) / len(
            poorly_formatted
        )
        print(f"  • Poorly formatted docs: {avg_poor:+.1f}% average improvement")

    if well_formatted:
        avg_good = sum(r["completeness_improvement"] for r in well_formatted) / len(
            well_formatted
        )
        print(f"  • Well formatted docs: {avg_good:+.1f}% average improvement")

    if poorly_formatted and well_formatted and avg_poor > avg_good:
        print("  ✅ Refinement helps more with challenging documents")

    # Count significant improvements
    significant_improvements = sum(
        1 for r in results if r["completeness_improvement"] > 10
    )
    print(
        f"  • {significant_improvements}/{len(results)} documents had major improvements (>10%)"
    )

    # When not to use refinement
    minimal_improvements = sum(1 for r in results if r["completeness_improvement"] < 2)
    if minimal_improvements > 0:
        print(
            f"  ⚠️ {minimal_improvements}/{len(results)} documents showed minimal improvement"
        )
        print("    (Refinement may not be worth the cost for these)")

    print()
    print("🎯 Recommendations:")
    print("  • Use refinement for challenging, poorly-formatted, or critical documents")
    print("  • Consider skipping refinement for simple, well-structured text")
    print("  • Always test on your specific document types")
    print("  • Monitor both completeness AND confidence improvements")
    print("  • Factor in 2-5x cost increase when deciding")


if __name__ == "__main__":
    main()
