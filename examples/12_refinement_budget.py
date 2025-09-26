#!/usr/bin/env python3
"""
Example 12: Refinement Budget Controls
======================================

This example demonstrates how to use budget controls to manage costs
and performance when using refinement. Learn how to:

- Set budget limits for calls and tokens
- Handle budget exceeded scenarios gracefully
- Balance accuracy vs cost for different use cases
- Monitor budget usage in real-time
- Implement cost-effective refinement strategies

Essential for production deployments where cost control is critical.
"""

import time
import warnings
from typing import List, Optional

from pydantic import BaseModel, Field

from langstruct import Budget, LangStruct, Refine, RefinementStrategy

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BusinessDocumentSchema(BaseModel):
    """Schema for business document extraction"""

    document_type: str = Field(description="Type of business document")
    company_name: str = Field(description="Name of the company")
    date: str = Field(description="Document date in YYYY-MM-DD format")
    amount: Optional[float] = Field(description="Monetary amount if present")
    reference_number: Optional[str] = Field(description="Reference or ID number")
    parties_involved: Optional[List[str]] = Field(
        description="People or companies mentioned"
    )
    key_terms: Optional[List[str]] = Field(description="Important terms or conditions")


def estimate_token_usage(
    text: str, n_candidates: int = 3, refine_steps: int = 1
) -> int:
    """Rough estimation of token usage for refinement"""
    # Very rough approximation - actual usage may vary significantly
    base_tokens = len(text.split()) * 1.3  # Rough text tokenization
    schema_tokens = 200  # Approximate schema overhead

    # Candidate generation
    candidate_tokens = (base_tokens + schema_tokens) * n_candidates

    # Refinement steps
    refine_tokens = (base_tokens + schema_tokens) * refine_steps

    # Judge scoring
    judge_tokens = candidate_tokens * 0.3  # Judge reviews all candidates

    total = int(candidate_tokens + refine_tokens + judge_tokens)
    return total


def run_budget_test(
    extractor: LangStruct, text: str, config: dict, test_name: str
) -> dict:
    """Run a refinement test with budget tracking"""
    print(f"🧪 Test: {test_name}")
    print(f"   Config: {config}")

    start_time = time.time()

    try:
        result = extractor.extract(text, refine=config)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)

    end_time = time.time()
    duration = end_time - start_time

    test_result = {
        "test_name": test_name,
        "config": config,
        "success": success,
        "error": error,
        "duration": duration,
        "result": result,
    }

    if success and result:
        # Extract budget usage
        budget_used = result.metadata.get("refinement_budget_used", {})
        test_result.update(
            {
                "calls_used": budget_used.get("calls", 0),
                "tokens_used": budget_used.get("tokens", 0),
                "confidence": result.confidence,
                "completeness": sum(1 for v in result.entities.values() if v)
                / len(result.entities),
            }
        )

        print(f"   ✅ Success in {duration:.1f}s")
        print(f"   📊 Calls used: {budget_used.get('calls', 0)}")
        print(f"   📊 Estimated tokens: {budget_used.get('tokens', 'N/A')}")
        print(f"   💯 Confidence: {result.confidence:.1%}")
        print(
            f"   📈 Fields filled: {sum(1 for v in result.entities.values() if v)}/{len(result.entities)}"
        )

        if result.metadata.get("refinement_applied"):
            print(f"   🎯 Strategy: {result.metadata.get('refinement_strategy')}")
            print(f"   🔄 Candidates: {result.metadata.get('candidates_generated', 0)}")
            print(f"   ⚙️ Refine steps: {result.metadata.get('refinement_steps', 0)}")

    else:
        print(f"   ❌ Failed: {error}")

    print()
    return test_result


def main():
    print("💰 Refinement Budget Controls")
    print("=" * 50)

    # Complex business document that benefits from refinement
    business_text = """
    MEMORANDUM OF UNDERSTANDING

    Between: GlobalTech Industries Inc. and Innovation Partners LLC
    Date: March 15, 2024
    Reference: MOU-2024-GT-001

    This memorandum outlines the terms for a strategic partnership
    valued at approximately $2.5M over 18 months.

    Key participants:
    - Sarah Chen (CEO, GlobalTech)
    - Marcus Williams (Director, Innovation Partners)
    - Jennifer Rodriguez (Legal Counsel)

    Critical terms:
    - Exclusive licensing agreement for AI technologies
    - Revenue sharing: 70/30 split
    - Quarterly performance reviews required
    - Automatic renewal clause included
    """

    print("📄 Business Document:")
    print(business_text)
    print(f"📏 Estimated base tokens: ~{len(business_text.split()) * 1.3:.0f}")
    print()

    extractor = LangStruct(schema=BusinessDocumentSchema)

    # 1. Budget Limit Testing
    print("1️⃣  BUDGET LIMIT TESTING")
    print("-" * 50)

    budget_tests = [
        {
            "name": "No Budget (Unlimited)",
            "config": {
                "strategy": "bon_then_refine",
                "n_candidates": 5,
                "max_refine_steps": 3,
            },
        },
        {
            "name": "Generous Budget",
            "config": {
                "strategy": "bon_then_refine",
                "n_candidates": 5,
                "max_refine_steps": 3,
                "budget": Budget(max_calls=15, max_tokens=20000),
            },
        },
        {
            "name": "Moderate Budget",
            "config": {
                "strategy": "bon_then_refine",
                "n_candidates": 3,
                "max_refine_steps": 2,
                "budget": Budget(max_calls=8, max_tokens=10000),
            },
        },
        {
            "name": "Strict Budget",
            "config": {
                "strategy": "bon",  # Only Best-of-N to save tokens
                "n_candidates": 3,
                "budget": Budget(max_calls=5, max_tokens=5000),
            },
        },
        {
            "name": "Very Strict Budget",
            "config": {
                "strategy": "bon",
                "n_candidates": 2,
                "budget": Budget(max_calls=3, max_tokens=2000),
            },
        },
    ]

    test_results = []

    for test in budget_tests:
        result = run_budget_test(extractor, business_text, test["config"], test["name"])
        test_results.append(result)

    # 2. Cost-Effectiveness Analysis
    print("2️⃣  COST-EFFECTIVENESS ANALYSIS")
    print("-" * 50)

    successful_tests = [r for r in test_results if r["success"]]

    if successful_tests:
        print("📊 Budget vs Performance Analysis:")
        print()

        for result in successful_tests:
            if "confidence" in result:
                calls = result.get("calls_used", 0)
                completeness = result.get("completeness", 0) * 100
                confidence = result.get("confidence", 0) * 100
                duration = result.get("duration", 0)

                # Calculate efficiency metrics
                confidence_per_call = confidence / calls if calls > 0 else 0
                completeness_per_call = completeness / calls if calls > 0 else 0

                print(f"🔹 {result['test_name']}")
                print(f"   Calls: {calls}, Duration: {duration:.1f}s")
                print(
                    f"   Confidence: {confidence:.1f}%, Completeness: {completeness:.1f}%"
                )
                print(
                    f"   Efficiency: {confidence_per_call:.1f} conf/call, {completeness_per_call:.1f} comp/call"
                )
                print()

    # 3. Real-world Budget Strategies
    print("3️⃣  REAL-WORLD BUDGET STRATEGIES")
    print("-" * 50)

    # Different strategies for different use cases
    strategies = {
        "Development/Testing": {
            "strategy": "bon",
            "n_candidates": 2,
            "budget": Budget(max_calls=3, max_tokens=3000),
            "description": "Fast iteration, minimal cost",
        },
        "Production (Cost-Conscious)": {
            "strategy": "bon",
            "n_candidates": 3,
            "budget": Budget(max_calls=5, max_tokens=7000),
            "description": "Balanced accuracy and cost",
        },
        "Production (Quality-First)": {
            "strategy": "bon_then_refine",
            "n_candidates": 4,
            "max_refine_steps": 2,
            "budget": Budget(max_calls=10, max_tokens=15000),
            "description": "Maximum accuracy within reason",
        },
        "High-Value Documents": {
            "strategy": "bon_then_refine",
            "n_candidates": 5,
            "max_refine_steps": 3,
            "budget": Budget(max_calls=20, max_tokens=25000),
            "description": "Spare no expense for critical docs",
        },
    }

    print("💡 Recommended Budget Strategies:")
    print()

    strategy_results = {}

    for use_case, config in strategies.items():
        description = config.pop("description")

        print(f"📋 {use_case}")
        print(f"   Purpose: {description}")

        # Estimate cost
        estimated_tokens = estimate_token_usage(
            business_text,
            config.get("n_candidates", 3),
            config.get("max_refine_steps", 1),
        )

        print(f"   Estimated tokens: {estimated_tokens:,}")
        print(
            f"   Budget: {config['budget'].max_calls} calls, {config['budget'].max_tokens:,} tokens"
        )

        # Test the strategy
        result = run_budget_test(extractor, business_text, config, use_case)
        strategy_results[use_case] = result

        # Add description back for display
        config["description"] = description

    # 4. Batch Processing Budget Planning
    print("4️⃣  BATCH PROCESSING BUDGET PLANNING")
    print("-" * 50)

    # Simulate batch processing multiple documents
    batch_size = 10
    print(f"📦 Planning budget for {batch_size} documents:")
    print()

    for use_case, config in strategies.items():
        if use_case in strategy_results and strategy_results[use_case]["success"]:
            result = strategy_results[use_case]

            calls_per_doc = result.get("calls_used", 0)
            tokens_per_doc = result.get("tokens_used", 0)

            # Estimate batch totals
            batch_calls = calls_per_doc * batch_size
            batch_tokens = tokens_per_doc * batch_size
            batch_duration = (
                result.get("duration", 0) * batch_size / 3
            )  # Assume some parallelization

            print(f"🔹 {use_case} for {batch_size} docs:")
            print(f"   Total calls: {batch_calls}")
            print(f"   Total tokens: {batch_tokens:,}")
            print(
                f"   Estimated time: {batch_duration:.1f}s ({batch_duration/60:.1f}m)"
            )
            print()

    # 5. Dynamic Budget Adjustment
    print("5️⃣  DYNAMIC BUDGET ADJUSTMENT")
    print("-" * 50)

    print("💡 Smart Budget Management Tips:")
    print()

    print("🎯 Document Complexity Assessment:")
    print("   • Simple docs (invoices): Budget(max_calls=3, max_tokens=3000)")
    print("   • Medium docs (contracts): Budget(max_calls=6, max_tokens=8000)")
    print("   • Complex docs (legal): Budget(max_calls=12, max_tokens=15000)")
    print()

    print("⚡ Performance vs Cost Trade-offs:")
    print("   • Development: Prioritize speed - use 'bon' strategy only")
    print("   • Production: Balance quality/cost - moderate budgets")
    print("   • Critical docs: Quality first - generous budgets")
    print()

    print("📊 Monitoring and Alerts:")
    print("   • Track calls_used and tokens_used in metadata")
    print("   • Set up alerts when budget utilization > 80%")
    print("   • Log budget exceeded events for analysis")
    print()

    print("🔄 Adaptive Strategies:")

    # Example of adaptive budget logic
    print("   ```python")
    print("   def adaptive_refinement(text, doc_importance):")
    print("       if doc_importance == 'critical':")
    print("           budget = Budget(max_calls=15, max_tokens=20000)")
    print("           strategy = 'bon_then_refine'")
    print("       elif doc_importance == 'standard':")
    print("           budget = Budget(max_calls=6, max_tokens=8000)")
    print("           strategy = 'bon'")
    print("       else:  # low importance")
    print("           budget = Budget(max_calls=3, max_tokens=3000)")
    print("           strategy = 'bon'")
    print("       ```")

    print()
    print("✨ Key Takeaways:")
    print("  • Always set budget limits in production")
    print("  • Monitor budget usage to optimize costs")
    print("  • Adjust strategy based on document importance")
    print("  • Test budget limits with your actual documents")
    print("  • Consider batch processing efficiency")
    print("  • Build alerting for budget overruns")


if __name__ == "__main__":
    main()
