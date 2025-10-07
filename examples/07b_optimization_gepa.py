#!/usr/bin/env python3
"""
LangStruct GEPA Optimization Example

Shows how to use GEPA optimizer for reflective prompt optimization with feedback.
GEPA is particularly effective for complex reasoning tasks where understanding
*why* extractions succeed or fail helps improve performance.

This example uses a two-model strategy:
- Gemini 2.5 Flash Lite: Fast model for actual extractions (cost-effective)
- Gemini 2.5 Flash: Stronger model for reflections (smart improvements)

Requirements:
    pip install langstruct

Environment:
    export GOOGLE_API_KEY="your-key"

Usage:
    python 07b_optimization_gepa.py
"""

import os
import warnings

from langstruct import LangStruct


def main():
    """Example showing GEPA optimization capabilities."""

    print("⚡ LangStruct GEPA Optimization Example")
    print("=" * 50)

    try:
        # Step 1: Create extractor with GEPA optimizer
        print()
        print("1️⃣ Creating extractor with GEPA optimizer...")
        print("   • Main model: Gemini 2.5 Flash Lite (fast, efficient)")
        print("   • Reflection model: Gemini 2.5 Flash (stronger reasoning)")
        print()

        import dspy

        from langstruct.optimizers import GEPAOptimizer

        # Create extractor with Gemini 2.5 Flash Lite as main model
        # Use google/ prefix to route to Google AI Studio (uses GOOGLE_API_KEY)
        extractor = LangStruct(
            example={
                "person_name": "Dr. Sarah Johnson",
                "job_title": "cardiologist",
                "years_experience": 8,
                "specialization": "interventional cardiology",
            },
            model="gemini/gemini-2.5-flash-lite",  # Fast model for extractions
            optimizer="gepa",
        )

        # Configure GEPA with stronger reflection model
        # The reflection model does the "thinking" about how to improve prompts
        # Important: Reflection needs higher max_tokens for detailed analysis
        gepa_optimizer = GEPAOptimizer(
            auto="light",  # Use "heavy" for production
            reflection_lm=dspy.LM(
                "gemini/gemini-2.5-flash",
                max_tokens=32000,  # High limit for detailed reflections
                temperature=1.0,  # Higher temp for creative improvements
            ),
            num_threads=4,
            track_stats=True,
        )
        extractor.optimizer = gepa_optimizer

        print("✅ Extractor ready with GEPA optimizer!")
        print("   • Gemini 2.5 Flash Lite: Handles actual extractions (fast)")
        print(
            "   • Gemini 2.5 Flash: Reflects on failures and proposes improvements (smart)"
        )
        print("   • Reflection model configured with:")
        print("     - max_tokens=32000 (needs space for detailed analysis)")
        print("     - temperature=1.0 (creativity for prompt improvements)")
        print()
        print("📚 About GEPA:")
        print("   • Uses reflective prompt evolution with textual feedback")
        print("   • The reflection model learns from *why* extractions succeed or fail")
        print("   • Best for complex reasoning tasks")
        print("   • Particularly effective when you have ground truth data")
        print("   • Requires high max_tokens for reflection LM (32000 recommended)")

        # Step 2: Initial extraction (baseline)
        print("\n2️⃣ Baseline extraction...")
        baseline_text = """
        Dr. Michael Chen is a senior software engineer with 12 years of experience
        specializing in machine learning and distributed systems. He currently leads
        the AI infrastructure team at TechCorp.
        """

        baseline_result = extractor.extract(baseline_text)
        print(f"   Baseline confidence: {baseline_result.confidence:.1%}")
        print(f"   Extracted: {baseline_result.entities}")

        # Step 3: Prepare training data for GEPA optimization
        print("\n3️⃣ Preparing training data for GEPA...")
        print("   Note: GEPA works best WITH expected results (ground truth)")

        # Training texts - varied examples from your domain
        training_texts = [
            "Dr. Lisa Wang is a pediatrician with 6 years of experience specializing in neonatal care.",
            "Prof. James Miller, an experienced biochemist (15+ years), focuses on protein structure research.",
            "Sarah Kim works as a data scientist for 4 years, specializing in natural language processing.",
            "Dr. Robert Taylor is a neurologist with 10 years experience in epilepsy treatment.",
            "Emily Rodriguez is a software architect with 8 years focusing on cloud infrastructure.",
        ]

        # Expected results - GEPA uses these for rich feedback
        expected_results = [
            {
                "person_name": "Dr. Lisa Wang",
                "job_title": "pediatrician",
                "years_experience": 6,
                "specialization": "neonatal care",
            },
            {
                "person_name": "Prof. James Miller",
                "job_title": "biochemist",
                "years_experience": 15,
                "specialization": "protein structure research",
            },
            {
                "person_name": "Sarah Kim",
                "job_title": "data scientist",
                "years_experience": 4,
                "specialization": "natural language processing",
            },
            {
                "person_name": "Dr. Robert Taylor",
                "job_title": "neurologist",
                "years_experience": 10,
                "specialization": "epilepsy treatment",
            },
            {
                "person_name": "Emily Rodriguez",
                "job_title": "software architect",
                "years_experience": 8,
                "specialization": "cloud infrastructure",
            },
        ]

        print(f"   📚 Training set: {len(training_texts)} examples")
        print(f"   🎯 With ground truth for feedback generation")

        # Step 4: Run GEPA optimization
        print("\n4️⃣ Running GEPA optimization...")
        print("   GEPA will:")
        print("   • Generate detailed feedback on extraction quality")
        print("   • Identify which fields were missed or incorrect")
        print("   • Suggest improvements to prompts")
        print("   • Evolve prompts based on feedback")
        print()

        # Define a small test set to evaluate before/after
        test_texts = [
            "Dr. Amanda Foster is a radiologist with 7 years of experience in diagnostic imaging.",
            "Mark Thompson works as a DevOps engineer for 5 years, specializing in Kubernetes orchestration.",
        ]
        expected_test_results = [
            {
                "person_name": "Dr. Amanda Foster",
                "job_title": "radiologist",
                "years_experience": 7,
                "specialization": "diagnostic imaging",
            },
            {
                "person_name": "Mark Thompson",
                "job_title": "DevOps engineer",
                "years_experience": 5,
                "specialization": "Kubernetes orchestration",
            },
        ]

        # Evaluate baseline performance
        baseline_scores = extractor.evaluate(test_texts, expected_test_results)
        print(
            f"   📉 Baseline — Accuracy: {baseline_scores.get('accuracy', 0):.1%}, "
            f"F1: {baseline_scores.get('f1', 0):.3f}"
        )

        did_optimize = False
        try:
            # Run GEPA optimization (requires API key)
            if (
                os.getenv("LANGSTRUCT_REAL_OPT")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
            ):
                print("   ⏳ GEPA is analyzing extractions and generating feedback...")
                print("   (This will take a few minutes)")

                extractor.optimize(
                    texts=training_texts,
                    expected_results=expected_results,
                )
                did_optimize = True
                print("   ✅ GEPA optimization complete!")
                print("   📈 Prompts evolved based on extraction feedback")
            else:
                print("   ⏳ GEPA optimization (simulated for demo)")
                print("   ✅ Optimization complete! (simulated)")
                print(
                    "   📈 In real use: GEPA evolves prompts based on detailed feedback"
                )
        except Exception as e:
            warnings.warn(f"Optimization failed or was skipped: {e}")

        # Evaluate post-optimization performance
        post_scores = extractor.evaluate(test_texts, expected_test_results)
        print(
            f"   📈 Post-GEPA — Accuracy: {post_scores.get('accuracy', 0):.1%}, "
            f"F1: {post_scores.get('f1', 0):.3f}"
        )

        if did_optimize:
            delta_acc = post_scores.get("accuracy", 0) - baseline_scores.get(
                "accuracy", 0
            )
            delta_f1 = post_scores.get("f1", 0) - baseline_scores.get("f1", 0)
            print(f"   Δ Improvement — Accuracy: {delta_acc:+.1%}, F1: {delta_f1:+.3f}")

        # Step 5: Test optimized performance
        print("\n5️⃣ Testing GEPA-optimized performance...")

        for i, test_text in enumerate(test_texts, 1):
            result = extractor.extract(test_text)
            print(f"\n   Test {i}:")
            print(f"   ├─ Confidence: {result.confidence:.1%}")
            print(f"   ├─ Name: {result.entities.get('person_name')}")
            print(f"   ├─ Title: {result.entities.get('job_title')}")
            print(f"   ├─ Experience: {result.entities.get('years_experience')} years")
            print(f"   └─ Specialization: {result.entities.get('specialization')}")

        # Step 6: GEPA vs MIPROv2 - When to use which?
        print("\n6️⃣ GEPA vs MIPROv2 - When to use which?")
        print()
        print("   Use GEPA when:")
        print("   ✓ You have ground truth data for training")
        print("   ✓ The task requires complex reasoning")
        print("   ✓ Understanding *why* extractions fail is valuable")
        print("   ✓ You want the optimizer to learn from detailed feedback")
        print("   ✓ You need Pareto-optimal prompt evolution")
        print()
        print("   Use MIPROv2 when:")
        print("   ✓ You want fast, general-purpose optimization")
        print("   ✓ The task is relatively straightforward")
        print("   ✓ You have limited training data")
        print("   ✓ Joint instruction + example optimization is sufficient")
        print()
        print("   💡 Pro tip: Try both and see which works better for your use case!")

        # Step 7: GEPA-specific features
        print("\n7️⃣ GEPA-Specific Features:")
        print("   • Textual feedback generation (not just scores)")
        print("   • Reflective prompt evolution (learns from mistakes)")
        print("   • Pareto frontier candidate selection")
        print("   • Can merge successful program variants")
        print("   • Supports inference-time search")
        print("   • Detailed optimization statistics tracking")

        # Step 8: Best practices
        print("\n8️⃣ GEPA Optimization Best Practices:")
        print("   • Always provide expected results for best feedback")
        print("   • Use 'auto=heavy' for production (we used 'light' for demo)")
        print()
        print("   • Two-model strategy (like this example):")
        print("     - Fast model (Gemini 2.5 Flash Lite) for actual extractions")
        print("     - Strong model (Gemini 2.5 Flash) for reflections")
        print("     This balances cost & quality!")
        print()
        print("   • Example configuration:")
        print("     from langstruct.optimizers import GEPAOptimizer")
        print("     import dspy")
        print()
        print("     optimizer = GEPAOptimizer(")
        print("         reflection_lm=dspy.LM(")
        print("             'gemini/gemini-2.5-flash',")
        print("             max_tokens=32000,  # Important for detailed reflections!")
        print("             temperature=1.0    # Higher temp for creative improvements")
        print("         ),")
        print("         auto='heavy'")
        print("     )")
        print()
        print("   • Start with 20-50 diverse training examples")
        print("   • Monitor the feedback to understand what GEPA learns")
        print("   • Use track_stats=True to see optimization progress")

        print("\n🎉 GEPA optimization provides richer feedback-driven improvement!")
        print("   Reflective evolution = better prompts over time")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Common fixes:")
        print("   • Ensure GOOGLE_API_KEY is set: export GOOGLE_API_KEY='your-key'")
        print("   • Check that API key has sufficient quota")
        print("   • Verify Gemini 2.0 Flash models are available in your region")
        print("   • Check that training data format is correct")
        print("   • Verify network stability during optimization")


if __name__ == "__main__":
    main()
