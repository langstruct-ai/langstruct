#!/usr/bin/env python3
"""
LangStruct Error Handling Example

Production-ready patterns for handling errors gracefully.

Requirements:
    pip install langstruct

Environment:
    GOOGLE_API_KEY or OPENAI_API_KEY required

Usage:
    python 05_error_handling.py
"""

import warnings
from typing import List, Optional

from langstruct import ExtractionError, LangStruct, ValidationError


class RobustExtractor:
    """Production-ready extractor with comprehensive error handling."""

    def __init__(self, example_schema, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        try:
            self.extractor = LangStruct(example=example_schema)
            print("✅ Extractor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize extractor: {e}")
            raise

    def safe_extract(self, text: str, max_retries: int = 2) -> Optional[dict]:
        """Extract with error handling and retries."""

        if not text or not text.strip():
            print("⚠️  Warning: Empty text provided")
            return None

        for attempt in range(max_retries + 1):
            try:
                print(f"🔄 Extraction attempt {attempt + 1}...")

                result = self.extractor.extract(text)

                # Check confidence threshold
                if result.confidence < self.min_confidence:
                    print(
                        f"⚠️  Low confidence: {result.confidence:.1%} < {self.min_confidence:.1%}"
                    )
                    if attempt < max_retries:
                        print(f"   Retrying... ({attempt + 1}/{max_retries})")
                        continue
                    else:
                        print(f"   Proceeding with low confidence result")

                print(f"✅ Success! Confidence: {result.confidence:.1%}")
                return {
                    "entities": result.entities,
                    "confidence": result.confidence,
                    "sources": result.sources if hasattr(result, "sources") else None,
                }

            except ValidationError as e:
                print(f"❌ Validation error: {e}")
                if attempt < max_retries:
                    print(f"   Retrying with adjusted parameters...")
                    continue
                else:
                    print(f"   All retries exhausted")
                    return None

            except ExtractionError as e:
                print(f"❌ Extraction error: {e}")
                if attempt < max_retries:
                    print(f"   Retrying...")
                    continue
                else:
                    print(f"   All retries exhausted")
                    return None

            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                if attempt < max_retries:
                    print(f"   Retrying...")
                    continue
                else:
                    print(f"   All retries exhausted")
                    return None

        return None

    def batch_extract_safe(self, texts: List[str]) -> List[dict]:
        """Process multiple texts with individual error handling."""
        results = []

        for i, text in enumerate(texts, 1):
            print(f"\n📄 Processing document {i}/{len(texts)}...")

            try:
                result = self.safe_extract(text)
                if result:
                    result["document_id"] = i
                    results.append(result)
                else:
                    print(f"   ⚠️  Document {i} failed extraction")
                    # Add placeholder for failed extraction
                    results.append(
                        {
                            "document_id": i,
                            "entities": {},
                            "confidence": 0.0,
                            "error": "Extraction failed",
                            "sources": None,
                        }
                    )

            except Exception as e:
                print(f"   ❌ Document {i} error: {e}")
                results.append(
                    {
                        "document_id": i,
                        "entities": {},
                        "confidence": 0.0,
                        "error": str(e),
                        "sources": None,
                    }
                )

        return results


def main():
    """Demonstrate production-ready error handling."""

    print("🛡️  LangStruct Error Handling Example")
    print("=" * 40)

    # Test data including problematic cases
    test_documents = [
        "Apple Inc. reported Q3 2024 revenue of $125.3 billion, up 15% year-over-year.",  # Good
        "",  # Empty text
        "Random text with no extractable information whatsoever.",  # Low quality
        "Microsoft Corporation announced $62.9 billion in Q3 2024 revenue.",  # Good
        "   ",  # Whitespace only
        "Tesla Q3 revenue: $25.2B (+9% YoY) - automotive segment strong.",  # Good but informal
    ]

    try:
        # Step 1: Initialize robust extractor
        print("\n1️⃣ Initializing robust extractor...")

        robust_extractor = RobustExtractor(
            example_schema={
                "company": "Apple Inc.",
                "quarter": "Q3 2024",
                "revenue": "125.3 billion",
                "growth": "15%",
            },
            min_confidence=0.6,  # Lower threshold for demo
        )

        # Step 2: Process documents with error handling
        print("\n2️⃣ Processing documents with error handling...")
        results = robust_extractor.batch_extract_safe(test_documents)

        # Step 3: Analyze results
        print("\n3️⃣ Results Analysis:")
        successful = [r for r in results if r["confidence"] > 0.6]
        failed = [r for r in results if r["confidence"] <= 0.6]

        print(f"   • Total documents: {len(results)}")
        print(f"   • Successful extractions: {len(successful)}")
        print(f"   • Failed extractions: {len(failed)}")

        # Step 4: Show successful extractions
        print("\n4️⃣ Successful Extractions:")
        for result in successful:
            doc_id = result["document_id"]
            entities = result["entities"]
            confidence = result["confidence"]
            print(f"   Document {doc_id}: {entities} (confidence: {confidence:.1%})")

        # Step 5: Show failed extractions
        print("\n5️⃣ Failed Extractions:")
        for result in failed:
            doc_id = result["document_id"]
            error = result.get("error", "Low confidence")
            print(f"   Document {doc_id}: {error}")

        # Step 6: Production recommendations
        print("\n6️⃣ Production Best Practices:")
        print("   • Set appropriate confidence thresholds for your use case")
        print("   • Implement retry logic with exponential backoff")
        print("   • Log all failures for analysis and improvement")
        print("   • Use circuit breakers for external API calls")
        print("   • Monitor extraction success rates and confidence scores")
        print("   • Have fallback strategies for critical extractions")

        print("\n🎉 Robust error handling ensures production reliability!")
        print("   Next: Try example 06_rag_integration.py for RAG enhancement")

    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("\n💡 Emergency checklist:")
        print("   • Verify API key is set and valid")
        print("   • Check internet connectivity")
        print("   • Ensure all dependencies are installed")
        print("   • Review logs for specific error details")


if __name__ == "__main__":
    main()
