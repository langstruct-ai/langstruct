#!/usr/bin/env python3
"""
Example 10: Advanced Refinement Configuration
=============================================

This example demonstrates advanced refinement features including:
- Different refinement strategies (BON, refine, BON+refine)
- Custom judge rubrics for domain-specific scoring
- Budget controls to manage cost and performance
- Refinement trace analysis for debugging

Learn how to fine-tune refinement for your specific use case.
"""

import time
import warnings
from typing import List, Optional

from pydantic import BaseModel

from langstruct import Budget, Field, LangStruct, Refine, RefinementStrategy

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class MedicalRecordSchema(BaseModel):
    """Schema for medical record extraction"""

    patient_name: str = Field(description="Full patient name")
    age: Optional[int] = Field(description="Patient age in years")
    diagnosis: str = Field(description="Primary medical diagnosis")
    medications: Optional[List[str]] = Field(
        description="List of prescribed medications"
    )
    physician: str = Field(description="Attending physician name")
    visit_date: Optional[str] = Field(description="Date of visit in YYYY-MM-DD format")


def main():
    print("⚙️ Advanced Refinement Configuration")
    print("=" * 50)

    # Complex medical record (challenging for extraction)
    medical_text = """
    Patient: Smith, John D.
    DOB: 1978-05-15 (Age 45)
    Visit Date: 2024-01-20

    Chief Complaint: Patient presents with chest pain and shortness of breath.

    Assessment: Acute myocardial infarction (STEMI)

    Medications prescribed:
    - Aspirin 81mg daily
    - Metoprolol 50mg twice daily
    - Atorvastatin 40mg at bedtime

    Attending: Dr. Sarah Johnson, MD
    Cardiology Department
    """

    print("🏥 Medical Record Text:")
    print(medical_text)
    print()

    # Create base extractor
    extractor = LangStruct(schema=MedicalRecordSchema)

    # 1. Different Refinement Strategies
    print("1️⃣  REFINEMENT STRATEGIES")
    print("-" * 40)

    strategies = {
        "Basic (no refinement)": None,
        "Best-of-N only": {"strategy": "bon", "n_candidates": 4},
        "Refinement only": {"strategy": "refine", "max_refine_steps": 2},
        "BON + Refinement": {
            "strategy": "bon_then_refine",
            "n_candidates": 3,
            "max_refine_steps": 2,
        },
    }

    results = {}

    for name, config in strategies.items():
        print(f"🔹 Testing: {name}")
        start_time = time.time()

        if config is None:
            result = extractor.extract(medical_text)
        else:
            result = extractor.extract(medical_text, refine=config)

        end_time = time.time()
        results[name] = result

        # Count filled fields
        filled_fields = sum(1 for v in result.entities.values() if v)
        total_fields = len(result.entities)

        print(f"  ✓ Filled fields: {filled_fields}/{total_fields}")
        print(f"  ✓ Confidence: {result.confidence:.1%}")
        print(f"  ✓ Time: {end_time - start_time:.1f}s")

        if result.metadata.get("refinement_applied"):
            print(f"  ✓ Candidates: {result.metadata.get('candidates_generated', 0)}")
            print(f"  ✓ Refine steps: {result.metadata.get('refinement_steps', 0)}")

        print()

    # 2. Custom Judge Rubrics
    print("2️⃣  CUSTOM JUDGE RUBRICS")
    print("-" * 40)

    # Medical-specific judge rubric
    medical_judge = """
    Score candidates based on medical accuracy:
    1. Patient names must be complete (First Last format)
    2. Ages should be exact numbers, not ranges
    3. Diagnoses must use proper medical terminology
    4. Medication names should be precise with dosages
    5. Physician names must include titles (Dr., MD, etc.)
    6. Dates should be in ISO format (YYYY-MM-DD)
    Penalize any hallucinated medical information not in the text.
    """

    print("🩺 Using medical-specific judge:")
    print(f"Judge rubric: {medical_judge[:100]}...")
    print()

    custom_result = extractor.extract(
        medical_text,
        refine={
            "strategy": "bon_then_refine",
            "n_candidates": 5,
            "judge": medical_judge,
            "max_refine_steps": 1,
        },
    )

    print("📊 Custom Judge Result:")
    for field, value in custom_result.entities.items():
        print(f"  {field}: {value}")
    print(f"💯 Confidence: {custom_result.confidence:.1%}")
    print()

    # 3. Budget Controls
    print("3️⃣  BUDGET CONTROLS")
    print("-" * 40)

    budget_configs = [
        {"name": "Unlimited", "budget": None},
        {"name": "Conservative", "budget": Budget(max_calls=5, max_tokens=10000)},
        {"name": "Strict", "budget": Budget(max_calls=3, max_tokens=5000)},
    ]

    for config in budget_configs:
        print(f"🔹 Testing: {config['name']} Budget")

        refine_config = {
            "strategy": "bon_then_refine",
            "n_candidates": 5,
            "max_refine_steps": 3,
        }

        if config["budget"]:
            refine_config["budget"] = config["budget"]

        result = extractor.extract(medical_text, refine=refine_config)

        # Show budget usage
        if result.metadata.get("refinement_budget_used"):
            budget_used = result.metadata["refinement_budget_used"]
            print(f"  📊 Calls used: {budget_used.get('calls', 0)}")
            print(f"  📊 Tokens used: {budget_used.get('tokens', 0)}")

        print(
            f"  ✓ Strategy applied: {result.metadata.get('refinement_strategy', 'None')}"
        )
        print(f"  ✓ Confidence: {result.confidence:.1%}")
        print()

    # 4. Using Refine Configuration Object
    print("4️⃣  REFINE CONFIGURATION OBJECT")
    print("-" * 40)

    # Create sophisticated configuration
    advanced_config = Refine(
        strategy=RefinementStrategy.BON_THEN_REFINE,
        n_candidates=4,
        judge="Focus on extracting complete patient information with exact medication dosages and proper medical terminology",
        max_refine_steps=2,
        temperature=0.8,  # Higher temperature for more diverse candidates
        budget=Budget(max_calls=8, max_tokens=15000),
    )

    print("⚙️ Advanced Refine Configuration:")
    print(f"  Strategy: {advanced_config.strategy}")
    print(f"  Candidates: {advanced_config.n_candidates}")
    print(f"  Max refine steps: {advanced_config.max_refine_steps}")
    print(f"  Temperature: {advanced_config.temperature}")
    print(
        f"  Budget: {advanced_config.budget.max_calls} calls, {advanced_config.budget.max_tokens} tokens"
    )
    print()

    # Can use at constructor level
    advanced_extractor = LangStruct(schema=MedicalRecordSchema, refine=advanced_config)

    advanced_result = advanced_extractor.extract(medical_text)

    print("📊 Advanced Configuration Result:")
    for field, value in advanced_result.entities.items():
        print(f"  {field}: {value}")
    print()

    # 5. Refinement Trace Analysis
    print("5️⃣  REFINEMENT TRACE ANALYSIS")
    print("-" * 40)

    # Get detailed trace information
    trace_result = extractor.extract(
        medical_text,
        refine={
            "strategy": "bon_then_refine",
            "n_candidates": 3,
            "max_refine_steps": 2,
        },
    )

    print("🔍 Refinement Trace:")
    trace = trace_result.metadata

    if trace.get("refinement_applied"):
        print(f"  🎯 Strategy: {trace.get('refinement_strategy')}")
        print(f"  📊 Candidates generated: {trace.get('candidates_generated')}")
        print(f"  🔄 Refinement steps taken: {trace.get('refinement_steps')}")

        budget_used = trace.get("refinement_budget_used", {})
        print(
            f"  💰 Budget used: {budget_used.get('calls', 0)} calls, {budget_used.get('tokens', 0)} tokens"
        )

        # Show if budget was exceeded
        if budget_used.get("calls", 0) >= 10:  # Assuming default budget
            print("  ⚠️  Budget limits may have been reached")
    else:
        print("  ❌ Refinement was not applied (check configuration)")

    print()

    # 6. Performance and Cost Analysis
    print("6️⃣  PERFORMANCE ANALYSIS")
    print("-" * 40)

    baseline = results["Basic (no refinement)"]
    best_refined = results["BON + Refinement"]

    # Compare field completeness
    baseline_filled = sum(1 for v in baseline.entities.values() if v)
    refined_filled = sum(1 for v in best_refined.entities.values() if v)
    total_fields = len(baseline.entities)

    improvement = ((refined_filled - baseline_filled) / total_fields) * 100
    confidence_gain = best_refined.confidence - baseline.confidence

    print("📈 Performance Summary:")
    print(f"  🎯 Field completion improvement: {improvement:+.1f}%")
    print(f"  💯 Confidence improvement: {confidence_gain:+.1%}")

    # Estimate cost impact (rough approximation)
    if best_refined.metadata.get("candidates_generated", 1) > 1:
        candidates = best_refined.metadata["candidates_generated"]
        refine_steps = best_refined.metadata.get("refinement_steps", 0)
        estimated_multiplier = candidates + refine_steps
        print(f"  💰 Estimated cost multiplier: ~{estimated_multiplier}x")

    print()
    print("✨ Key Takeaways:")
    print("  • BON+Refinement usually gives best accuracy but highest cost")
    print("  • Custom judges help with domain-specific requirements")
    print("  • Budget controls prevent runaway costs")
    print("  • Use Refine objects for complex configurations")
    print("  • Monitor refinement metadata to understand what's happening")
    print("  • Balance accuracy gains vs speed/cost for your use case")


if __name__ == "__main__":
    main()
