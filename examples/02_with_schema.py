#!/usr/bin/env python3
"""
LangStruct Custom Schema Example

Shows how to use custom schemas for more control over extraction.

Requirements:
    pip install langstruct

Environment:
    GOOGLE_API_KEY or OPENAI_API_KEY required

Usage:
    python 02_with_schema.py
"""

from pydantic import BaseModel, Field

from langstruct import LangStruct


class PersonSchema(BaseModel):
    """Custom schema for person information."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    location: str = Field(description="Current city and state/country")
    occupation: str = Field(description="Job title or profession")


def main():
    """Example using custom Pydantic schemas."""

    print("📝 LangStruct Custom Schema Example")
    print("=" * 40)

    try:
        # Step 1: Create extractor with custom schema
        print("\n1️⃣ Creating extractor with custom schema...")
        extractor = LangStruct(schema=PersonSchema)
        print("✅ Extractor created with PersonSchema!")

        # Step 2: Extract from more complex text
        print("\n2️⃣ Extracting from complex text...")
        text = """
        Dr. Sarah Johnson is a 34-year-old cardiologist working at Boston General Hospital.
        She completed her medical degree at Harvard and currently lives in Cambridge, Massachusetts.
        Sarah specializes in interventional cardiology and has been practicing for 8 years.
        """

        result = extractor.extract(text)

        # Step 3: Show structured results
        print("\n3️⃣ Structured Results:")
        print(f"   Name: {result.entities.get('name')}")
        print(f"   Age: {result.entities.get('age')}")
        print(f"   Location: {result.entities.get('location')}")
        print(f"   Occupation: {result.entities.get('occupation')}")
        print(f"   Confidence: {result.confidence:.1%}")

        # Step 4: Show schema benefits
        print("\n4️⃣ Schema Benefits:")
        print(f"   • Type safety: Age is {type(result.entities.get('age')).__name__}")
        print(f"   • Validation: All fields checked for correct types")
        print(f"   • Documentation: Each field has clear descriptions")

        # Step 5: Show source tracking (if available)
        if result.sources:
            print("\n5️⃣ Source Tracking:")
            for field, spans in result.sources.items():
                if spans:
                    span = spans[0]  # Show first source
                    print(
                        f"   • {field}: '{span.text}' (chars {span.start}-{span.end})"
                    )

        print("\n🎉 Custom schemas provide more control and validation!")
        print("   Next: Try example 03_source_tracking.py to dive deeper into sources")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Common fixes:")
        print("   • Set your API key: export GOOGLE_API_KEY='your-key'")
        print("   • Check that langstruct is installed: pip install langstruct")


if __name__ == "__main__":
    main()
