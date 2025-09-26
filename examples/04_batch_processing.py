#!/usr/bin/env python3
"""
LangStruct Batch Processing Example

Shows how to process multiple documents efficiently.

Requirements:
    pip install langstruct

Environment:
    GOOGLE_API_KEY or OPENAI_API_KEY required

Usage:
    python 04_batch_processing.py
"""

from langstruct import LangStruct


def main():
    """Example processing multiple documents at once."""

    print("📦 LangStruct Batch Processing Example")
    print("=" * 40)

    try:
        # Step 1: Create extractor for product information
        print("\n1️⃣ Creating extractor for product data...")
        extractor = LangStruct(
            example={
                "product_name": "MacBook Pro",
                "price": 2399,
                "brand": "Apple",
                "category": "laptop",
            }
        )
        print("✅ Product extractor created!")

        # Step 2: Prepare multiple documents
        print("\n2️⃣ Preparing multiple product descriptions...")
        documents = [
            "The new MacBook Pro 16-inch costs $2,399 and represents Apple's flagship laptop offering.",
            "Samsung Galaxy S24 Ultra smartphone is priced at $1,199 and features cutting-edge mobile technology.",
            "The Tesla Model 3 electric vehicle starts at $38,990 and has revolutionized the automotive industry.",
            "Sony PlayStation 5 gaming console retails for $499 and offers next-generation gaming experiences.",
            "The iPad Air from Apple is available for $599 and provides excellent tablet performance.",
        ]

        print(f"   📄 Processing {len(documents)} documents...")

        # Step 3: Process all documents at once
        print("\n3️⃣ Running batch extraction...")
        results = extractor.extract(documents)  # Pass list of texts
        print(f"✅ Processed {len(results)} documents!")

        # Step 4: Show results for each document
        print("\n4️⃣ Batch Results:")
        for i, result in enumerate(results, 1):
            print(f"\n   Document {i}:")
            print(f"   ├─ Product: {result.entities.get('product_name', 'Unknown')}")
            print(
                f"   ├─ Price: ${result.entities.get('price', 'Unknown'):,}"
                if result.entities.get("price")
                else f"   ├─ Price: Unknown"
            )
            print(f"   ├─ Brand: {result.entities.get('brand', 'Unknown')}")
            print(f"   ├─ Category: {result.entities.get('category', 'Unknown')}")
            print(f"   └─ Confidence: {result.confidence:.1%}")

        # Step 5: Show aggregate statistics
        print("\n5️⃣ Batch Statistics:")
        total_confidence = sum(r.confidence for r in results)
        avg_confidence = total_confidence / len(results)
        successful_extractions = sum(1 for r in results if r.confidence > 0.7)

        print(f"   • Total documents: {len(results)}")
        print(f"   • Average confidence: {avg_confidence:.1%}")
        print(f"   • High-confidence extractions (>70%): {successful_extractions}")

        # Step 6: Show export options
        print("\n6️⃣ Export Options:")
        print("   • Save to CSV: extractor.export_batch(results, 'products.csv')")
        print(
            "   • Save to JSON: extractor.export_batch(results, 'products.json', format='json')"
        )
        print(
            "   • Save to Excel: extractor.export_batch(results, 'products.xlsx', format='excel')"
        )

        # Example export (uncomment to try):
        # extractor.export_batch(results, "products.csv")
        # print("   ✅ Exported to products.csv")

        print("\n🎉 Batch processing handles multiple documents efficiently!")
        print("   Next: Try example 05_error_handling.py for production patterns")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Common fixes:")
        print("   • Set your API key: export GOOGLE_API_KEY='your-key'")
        print("   • Check network connection")
        print("   • Verify langstruct installation: pip install langstruct")


if __name__ == "__main__":
    main()
