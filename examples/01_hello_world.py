#!/usr/bin/env python3
"""
LangStruct Hello World Example

The simplest possible LangStruct usage. Perfect for getting started.

Requirements:
    pip install langstruct

Environment:
    GOOGLE_API_KEY or OPENAI_API_KEY required

Usage:
    python 01_hello_world.py
"""

from langstruct import LangStruct


def main():
    """Simplest possible LangStruct example."""

    print("🌍 LangStruct Hello World Example")
    print("=" * 40)

    try:
        # Step 1: Create extractor from a simple example
        print("\n1️⃣ Creating extractor from example...")
        extractor = LangStruct(example={"name": "Alice", "age": 30})
        print("✅ Extractor created!")

        # Step 2: Extract from simple text
        print("\n2️⃣ Extracting from text...")
        text = "Hi, I'm Bob and I'm 25 years old."
        result = extractor.extract(text)

        # Step 3: Show results
        print("\n3️⃣ Results:")
        print(f"   Name: {result.entities.get('name', 'Not found')}")
        print(f"   Age: {result.entities.get('age', 'Not found')}")
        print(f"   Confidence: {result.confidence:.1%}")

        print("\n🎉 Success! That's all there is to it.")
        print("   Next: Try example 02_with_schema.py for more control")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Common fixes:")
        print("   • Set your API key: export GOOGLE_API_KEY='your-key'")
        print("   • Check your internet connection")
        print("   • Make sure langstruct is installed: pip install langstruct")


if __name__ == "__main__":
    main()
