#!/usr/bin/env python3
"""
LangStruct Save/Load Extractors Example

This example demonstrates how to save and load LangStruct extractors,
including optimized extractors and those with advanced configurations.

The save/load functionality allows you to:
- Save trained/optimized extractors for reuse
- Share extractors across different environments
- Version control your extraction pipelines
- Deploy extractors without retraining

Requirements:
    pip install langstruct

Environment:
    GOOGLE_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY required

Usage:
    python 13_save_load_extractors.py
"""

from pathlib import Path

from pydantic import BaseModel, Field

from langstruct import LangStruct
from langstruct.core.refinement import Refine


# Define a custom schema for our example
class ProductReviewSchema(BaseModel):
    """Schema for extracting product review information."""

    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(description="Rating score (1-5 stars)")
    sentiment: str = Field(description="Overall sentiment (positive/negative/neutral)")
    key_features: list[str] = Field(description="List of key features mentioned")
    reviewer_recommendation: bool = Field(
        description="Whether reviewer recommends the product"
    )


def demonstrate_basic_save_load():
    """Demonstrate basic save and load functionality."""
    print("🔧 Basic Save/Load Example")
    print("=" * 40)

    # Create an extractor with dynamic schema
    print("\n1️⃣ Creating extractor with dynamic schema...")
    extractor = LangStruct(
        example={
            "product_name": "iPhone 15",
            "rating": 4,
            "sentiment": "positive",
            "key_features": ["camera", "battery life"],
            "reviewer_recommendation": True,
        }
    )

    # Test the extractor
    sample_review = """
    I just got the new Samsung Galaxy S24 and I'm really impressed!
    The camera quality is outstanding, especially in low light.
    Battery easily lasts a full day of heavy use. The display is gorgeous.
    I'd definitely recommend this phone to anyone looking for an upgrade.
    Rating: 5 out of 5 stars.
    """

    print("2️⃣ Testing original extractor...")
    original_result = extractor.extract(sample_review)
    print(f"   Product: {original_result.entities.get('product_name')}")
    print(f"   Rating: {original_result.entities.get('rating')}")
    print(f"   Sentiment: {original_result.entities.get('sentiment')}")

    # Save the extractor
    print("\n3️⃣ Saving extractor...")
    save_path = Path("./saved_extractors/basic_review_extractor")
    extractor.save(str(save_path))
    print(f"   ✅ Saved to: {save_path}")

    # Load the extractor
    print("\n4️⃣ Loading extractor...")
    loaded_extractor = LangStruct.load(str(save_path))
    print("   ✅ Loaded successfully!")

    # Test loaded extractor
    print("\n5️⃣ Testing loaded extractor...")
    loaded_result = loaded_extractor.extract(sample_review)
    print(f"   Product: {loaded_result.entities.get('product_name')}")
    print(f"   Rating: {loaded_result.entities.get('rating')}")
    print(f"   Sentiment: {loaded_result.entities.get('sentiment')}")

    # Compare results
    print("\n6️⃣ Comparing results...")
    fields_match = set(original_result.entities.keys()) == set(
        loaded_result.entities.keys()
    )
    print(f"   Schema fields preserved: {'✅' if fields_match else '❌'}")

    return save_path


def demonstrate_predefined_schema_save_load():
    """Demonstrate save/load with predefined schema."""
    print("\n\n🎯 Predefined Schema Save/Load Example")
    print("=" * 45)

    # Create extractor with predefined schema
    print("\n1️⃣ Creating extractor with predefined schema...")
    extractor = LangStruct(schema=ProductReviewSchema)

    # Save extractor
    print("\n2️⃣ Saving extractor with predefined schema...")
    save_path = Path("./saved_extractors/predefined_schema_extractor")
    extractor.save(str(save_path))
    print(f"   ✅ Saved to: {save_path}")

    # Load extractor
    print("\n3️⃣ Loading extractor...")
    loaded_extractor = LangStruct.load(str(save_path))

    # Verify schema preservation
    print("\n4️⃣ Verifying schema preservation...")
    original_fields = extractor.schema.get_field_descriptions()
    loaded_fields = loaded_extractor.schema.get_field_descriptions()

    print(f"   Original schema: {extractor.schema.__name__}")
    print(f"   Loaded schema: {loaded_extractor.schema.__name__}")
    print(f"   Fields match: {'✅' if original_fields == loaded_fields else '❌'}")

    return save_path


def demonstrate_advanced_features_save_load():
    """Demonstrate save/load with advanced features like refinement."""
    print("\n\n⚡ Advanced Features Save/Load Example")
    print("=" * 42)

    # Create extractor with refinement (optimized for example speed)
    print("\n1️⃣ Creating extractor with refinement...")
    extractor = LangStruct(
        schema=ProductReviewSchema,
        refine=Refine(
            strategy="bon",  # Just Best-of-N for faster example
            n_candidates=2,  # Reduced from 3 to 2
            max_refine_steps=1,
        ),
        use_sources=True,
    )

    print("   ✅ Created with Best-of-N strategy (optimized for demo speed)")

    # Save extractor with advanced config
    print("\n2️⃣ Saving extractor with advanced configuration...")
    save_path = Path("./saved_extractors/advanced_extractor")
    extractor.save(str(save_path))

    # Check what files were created
    print("   📁 Saved files:")
    for file_path in save_path.iterdir():
        print(f"      - {file_path.name}")

    # Load extractor
    print("\n3️⃣ Loading advanced extractor...")
    loaded_extractor = LangStruct.load(str(save_path))

    # Verify advanced features are preserved
    print("\n4️⃣ Verifying advanced features...")
    original_refine = extractor.refine_config
    loaded_refine = loaded_extractor.refine_config

    if original_refine and loaded_refine:
        print(f"   Refinement strategy: {loaded_refine.strategy}")
        print(f"   Number of candidates: {loaded_refine.n_candidates}")
        print(f"   Max refine steps: {loaded_refine.max_refine_steps}")
        print("   ✅ Refinement configuration preserved!")
    else:
        print("   ❌ Refinement configuration not preserved")

    print(f"   Source grounding: {'✅' if loaded_extractor.use_sources else '❌'}")

    return save_path


def demonstrate_extraction_comparison():
    """Compare extraction quality before and after save/load."""
    print("\n\n📊 Extraction Quality Comparison")
    print("=" * 35)

    # Load the basic extractor
    basic_path = Path("./saved_extractors/basic_review_extractor")
    basic_extractor = LangStruct.load(str(basic_path))

    # Load the advanced extractor
    advanced_path = Path("./saved_extractors/advanced_extractor")
    advanced_extractor = LangStruct.load(str(advanced_path))

    # Test text
    test_review = """
    The MacBook Pro M3 is an absolute game-changer! The performance is incredible
    - I can run multiple heavy applications without any lag. The battery life
    is phenomenal, easily getting 12+ hours of real work. The display is sharp
    and vibrant. Build quality feels premium as always. My only complaint is the price,
    but for professional work, it's worth every penny. Would definitely buy again!
    4.5/5 stars - highly recommended!
    """

    print("\n🔍 Testing both extractors on the same text...")

    # Basic extraction
    print("\n📝 Basic Extractor Results:")
    basic_result = basic_extractor.extract(test_review)
    for field, value in basic_result.entities.items():
        print(f"   {field}: {value}")
    print(f"   Confidence: {basic_result.confidence:.2f}")

    # Advanced extraction (with refinement) - using simpler refinement to speed up example
    print("\n⚡ Advanced Extractor Results (with refinement):")
    advanced_result = advanced_extractor.extract(
        test_review,
        refine={
            "strategy": "bon",  # Just Best-of-N, no iterative refinement for speed
            "n_candidates": 2,  # Reduced from 3 to 2 candidates
        },
    )
    for field, value in advanced_result.entities.items():
        print(f"   {field}: {value}")
    print(f"   Confidence: {advanced_result.confidence:.2f}")

    # Show metadata if refinement was applied
    if advanced_result.metadata.get("refinement_applied"):
        print("   🔧 Refinement metadata:")
        print(f"      Strategy: {advanced_result.metadata.get('refinement_strategy')}")
        print(
            f"      Candidates: {advanced_result.metadata.get('candidates_generated')}"
        )
        print(f"      Steps: {advanced_result.metadata.get('refinement_steps')}")


def inspect_save_directory():
    """Inspect the contents of a saved extractor directory."""
    print("\n\n🔍 Save Directory Inspection")
    print("=" * 30)

    save_path = Path("./saved_extractors/advanced_extractor")

    print(f"\n📁 Contents of {save_path}:")

    for file_path in sorted(save_path.iterdir()):
        print(f"\n📄 {file_path.name}:")

        if file_path.suffix == ".json":
            import json

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if file_path.name == "langstruct_metadata.json":
                    print(f"   LangStruct Version: {data.get('langstruct_version')}")
                    print(f"   Schema Type: {data.get('schema_type')}")
                    print(f"   Model: {data.get('model_name')}")
                    print(
                        f"   Optimization Applied: {data.get('optimization_applied')}"
                    )
                    print(f"   Refinement Applied: {data.get('refinement_applied')}")
                elif file_path.name == "refinement_config.json":
                    print(f"   Strategy: {data.get('strategy')}")
                    print(f"   Candidates: {data.get('n_candidates')}")
                    print(f"   Max Steps: {data.get('max_refine_steps')}")
                else:
                    print(f"   Keys: {list(data.keys())[:5]}...")  # Show first 5 keys
            except Exception as e:
                print(f"   Error reading JSON: {e}")
        else:
            file_size = file_path.stat().st_size
            print(f"   Size: {file_size} bytes")


def cleanup_examples():
    """Clean up example save directories."""
    print("\n\n🧹 Cleanup")
    print("=" * 10)

    import shutil

    save_base = Path("./saved_extractors")

    if save_base.exists():
        print(f"Removing {save_base}...")
        shutil.rmtree(save_base)
        print("✅ Cleanup complete!")
    else:
        print("Nothing to clean up.")


def main():
    """Run all save/load examples."""
    print("💾 LangStruct Save/Load Extractors Example")
    print("=" * 50)
    print("\nThis example demonstrates the save/load functionality")
    print("for preserving and sharing LangStruct extractors.")
    print(
        "\n⏱️  Note: This example makes several LLM API calls and may take 2-5 minutes"
    )
    print("depending on your model and API response times.\n")

    try:
        # Run examples
        demonstrate_basic_save_load()
        demonstrate_predefined_schema_save_load()
        demonstrate_advanced_features_save_load()
        demonstrate_extraction_comparison()
        inspect_save_directory()

        print("\n\n🎉 All save/load examples completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • Extractors can be saved and loaded with full state preservation")
        print("   • Both dynamic and predefined schemas are supported")
        print("   • Advanced features like refinement are preserved")
        print("   • Save files are human-readable for debugging")
        print("   • API keys are never saved (security)")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Always cleanup
        cleanup_examples()


if __name__ == "__main__":
    main()
