#!/usr/bin/env python3
"""
LangStruct Source Tracking Example

Shows how to use source grounding to track where extracted data comes from.

Requirements:
    pip install langstruct

Environment:
    GOOGLE_API_KEY or OPENAI_API_KEY required

Usage:
    python 03_source_tracking.py
"""

from langstruct import LangStruct


def main():
    """Example demonstrating source tracking capabilities."""

    print("📍 LangStruct Source Tracking Example")
    print("=" * 40)

    try:
        # Step 1: Create extractor (source tracking enabled by default)
        print("\n1️⃣ Creating extractor with source tracking...")
        extractor = LangStruct(
            example={
                "company": "Apple Inc.",
                "revenue": "125.3 billion",
                "quarter": "Q3 2024",
                "growth": "15%",
            }
        )
        print("✅ Extractor created with source tracking enabled!")

        # Step 2: Extract from financial text
        print("\n2️⃣ Extracting from financial document...")
        text = """
        Apple Inc. reported stellar Q3 2024 results with record revenue of $125.3 billion,
        representing 15% year-over-year growth. The company's performance exceeded analyst
        expectations across all product categories, with iPhone sales leading the charge.
        """

        result = extractor.extract(text)

        # Step 3: Show extracted data
        print("\n3️⃣ Extracted Data:")
        for field, value in result.entities.items():
            print(f"   • {field}: {value}")
        print(f"   • Confidence: {result.confidence:.1%}")

        # Step 4: Show source locations
        print("\n4️⃣ Source Locations (where each piece came from):")
        if result.sources:
            for field, spans in result.sources.items():
                if spans:
                    for i, span in enumerate(spans):
                        print(f"   • {field}[{i}]: '{span.text}'")
                        print(f"     ↳ Found at characters {span.start}-{span.end}")
        else:
            print("   No source information available")

        # Step 5: Show why source tracking matters
        print("\n5️⃣ Why Source Tracking Matters:")
        print("   • Compliance: Audit trail for extracted data")
        print("   • Trust: Users can verify extractions against original text")
        print("   • Debugging: Find extraction issues quickly")
        print("   • RAG: Link extracted metadata back to source documents")

        # Step 6: Show original text with positions
        print("\n6️⃣ Original Text with Character Positions:")
        print("   (Numbers show character positions)")
        print()
        for i, char in enumerate(text):
            if i % 50 == 0:
                print(f"\n{i:3d}: ", end="")
            print(char, end="")
        print("\n")

        # Step 7: Generate interactive HTML visualization
        print("\n7️⃣ Interactive HTML Visualization:")
        print("   Generating visualization with source highlighting...")
        html_file = "source_tracking_results.html"
        extractor.visualize([result], html_file)
        print(f"   ✅ Saved to {html_file}")
        print("   Open this file in your browser to see:")
        print("   • Extracted fields highlighted in the original text")
        print("   • Interactive hover to see field names")
        print("   • Color-coded spans for each field")

        print("\n🎉 Source tracking provides full transparency!")
        print("   Next: Try example 04_batch_processing.py for multiple documents")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Common fixes:")
        print("   • Set your API key: export GOOGLE_API_KEY='your-key'")
        print("   • Ensure you have internet connectivity")


if __name__ == "__main__":
    main()
