#!/usr/bin/env python3
"""
LangStruct Query Parsing Example

Demonstrates bidirectional RAG enhancement with intelligent query parsing.
Shows how to parse natural language queries into structured filters automatically.

Requirements:
    pip install langstruct

Environment:
    GOOGLE_API_KEY or OPENAI_API_KEY required

Usage:
    python 08_query_parsing.py
"""

from langstruct import LangStruct, ParsedQuery


def demonstrate_basic_parsing():
    """Show basic query parsing capabilities."""

    print("🔍 Basic Query Parsing Demo")
    print("-" * 40)

    try:
        # Step 1: Create LangStruct instance with financial schema
        print("\n1️⃣ Creating LangStruct with financial schema...")

        schema_example = {
            "company": "Apple Inc.",
            "quarter": "Q3 2024",
            "revenue": 125.3,  # in billions
            "profit_margin": 23.1,  # percentage
            "growth_rate": 15.2,  # percentage
            "risks": ["Supply chain", "Regulatory"],
        }

        ls = LangStruct(example=schema_example)
        print("✅ LangStruct instance created!")

        # Step 2: Test various natural language queries
        print("\n2️⃣ Testing natural language queries...")

        test_queries = [
            "Show me Q3 2024 earnings reports",
            "Find tech companies with revenue over $100B",
            "Q3 profitable companies above 20% margins",
            "Large technology firms recent quarterly results",
            "Apple Microsoft Google Q3 2024 financial performance",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")

            try:
                result = ls.query(query)

                print(f"   ├─ Confidence: {result.confidence:.1%}")
                print(f"   ├─ Semantic terms: {result.semantic_terms}")
                print(f"   ├─ Structured filters: {result.structured_filters}")
                print(f"   └─ Explanation: {result.explanation}")

            except Exception as e:
                print(f"   ❌ Parsing failed: {e}")

        print("\n🎉 Basic parsing demonstration complete!")

    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_bidirectional_rag():
    """Show complete bidirectional RAG enhancement."""

    print("\n\n🚀 Bidirectional RAG Enhancement Demo")
    print("-" * 40)

    try:
        # Step 1: Set up document extraction side
        print("\n1️⃣ Setting up document extraction...")

        financial_schema = {
            "company_name": "Microsoft Corporation",
            "quarter": "Q3 2024",
            "revenue": 62.9,
            "revenue_growth": 17,
            "profit_margin": 38.1,
            "key_highlights": ["Azure growth", "AI integration"],
            "business_segments": [
                "Productivity",
                "Intelligence Cloud",
                "More Personal Computing",
            ],
        }

        # Single LangStruct instance for both extraction and query parsing
        langstruct = LangStruct(example=financial_schema)
        print("✅ LangStruct ready for both extraction and query parsing")

        # Step 3: Process sample documents
        print("\n3️⃣ Processing sample financial documents...")

        sample_documents = [
            """
            Microsoft Corporation announced strong Q3 2024 results with revenue of $62.9 billion,
            representing 17% year-over-year growth. The company's profit margin reached 38.1% 
            driven by Azure cloud services and AI integration across product lines. Key highlights
            included continued Azure growth and successful AI integration initiatives.
            """,
            """
            Apple Inc. reported record Q3 2024 revenue of $85.4 billion, up 15% from the prior year.
            The company achieved a 23.1% profit margin with strong performance across iPhone sales
            and Services segments. Leadership highlighted supply chain improvements and expanded
            global market presence.
            """,
            """
            Alphabet Inc. delivered Q3 2024 revenue of $76.7 billion, marking 11% year-over-year growth.
            Google's profit margin of 21.3% benefited from Search revenue strength and Cloud expansion.
            The company emphasized AI research advancements and YouTube advertising performance.
            """,
        ]

        # Extract metadata from documents
        extracted_metadata = []
        for i, doc in enumerate(sample_documents, 1):
            print(f"   Processing document {i}...")

            try:
                result = langstruct.extract(doc)
                extracted_metadata.append(
                    {
                        "doc_id": i,
                        "content": doc.strip(),
                        "metadata": result.entities,
                        "confidence": result.confidence,
                    }
                )
                print(
                    f"   ├─ Company: {result.entities.get('company_name', 'Unknown')}"
                )
                print(f"   ├─ Revenue: ${result.entities.get('revenue', 'Unknown')}B")
                print(f"   └─ Confidence: {result.confidence:.1%}")

            except Exception as e:
                print(f"   ❌ Extraction failed: {e}")

        print(f"\n   ✅ Extracted metadata from {len(extracted_metadata)} documents")

        # Step 4: Demonstrate intelligent query parsing
        print("\n4️⃣ Demonstrating intelligent query parsing...")

        user_queries = [
            "Show me Q3 2024 companies with revenue over $70B",
            "Find profitable tech giants with margins above 25%",
            "Q3 results from cloud-focused companies",
            "High-growth technology companies recent performance",
        ]

        for query in user_queries:
            print(f"\n   User Query: '{query}'")

            try:
                parsed = langstruct.query(query)

                print(f"   ├─ Semantic Search: {parsed.semantic_terms}")
                print(f"   ├─ Metadata Filters: {parsed.structured_filters}")
                print(f"   ├─ Confidence: {parsed.confidence:.1%}")

                # Simulate RAG retrieval with parsed filters
                matching_docs = []
                for doc_data in extracted_metadata:
                    metadata = doc_data["metadata"]
                    matches = True

                    # Apply parsed filters
                    for filter_key, filter_value in parsed.structured_filters.items():
                        if filter_key not in metadata:
                            continue

                        if isinstance(filter_value, dict) and "$gte" in filter_value:
                            threshold = filter_value["$gte"]
                            actual_value = metadata.get(filter_key, 0)
                            if (
                                isinstance(actual_value, (int, float))
                                and actual_value < threshold
                            ):
                                matches = False
                                break
                        elif metadata.get(filter_key) != filter_value:
                            matches = False
                            break

                    if matches:
                        matching_docs.append(doc_data)

                print(f"   └─ Matched Documents: {len(matching_docs)}")
                for doc in matching_docs[:2]:  # Show top 2
                    company = doc["metadata"].get("company_name", "Unknown")
                    revenue = doc["metadata"].get("revenue", "Unknown")
                    print(f"       • {company} (${revenue}B revenue)")

            except Exception as e:
                print(f"   ❌ Query parsing failed: {e}")

        # Step 5: Show the complete enhancement
        print("\n5️⃣ Bidirectional RAG Enhancement Summary:")
        print("   📄 Documents → LangStruct.extract() → Structured Metadata ✅")
        print("   🔍 Queries → LangStruct.query() → Structured Filters ✅")
        print("   🎯 Result: Precise RAG retrieval instead of fuzzy semantic search")
        print("   💡 Benefit: Users get exactly what they ask for")

        print("\n🎉 Bidirectional RAG enhancement demonstration complete!")

    except Exception as e:
        print(f"❌ Error in bidirectional demo: {e}")


def demonstrate_advanced_features():
    """Show advanced query parsing features."""

    print("\n\n⚡ Advanced Query Parsing Features")
    print("-" * 40)

    try:
        # Advanced parser with richer schema
        advanced_schema = {
            "company_name": "Tesla Inc.",
            "quarter": "Q3 2024",
            "revenue_billions": 25.2,
            "revenue_growth_pct": 9,
            "profit_margin_pct": 19.3,
            "vehicle_deliveries": 435000,
            "energy_revenue": 2.4,
            "geographic_segments": ["North America", "China", "Europe"],
            "key_risks": ["Supply chain", "Regulatory", "Competition"],
            "business_highlights": ["Model Y production", "Supercharger expansion"],
        }

        ls = LangStruct(example=advanced_schema)

        # Test advanced queries
        advanced_queries = [
            "Tesla Q3 deliveries and energy business performance",
            "EV companies with strong China presence above 400k deliveries",
            "Automotive firms recent margins over 15% growth above 5%",
        ]

        for query in advanced_queries:
            print(f"\n🔍 Advanced Query: '{query}'")

            # Full parsing with explanation
            result = ls.query(query, explain=True)
            print(f"Confidence: {result.confidence:.1%}")
            print("Explanation:")
            print(result.explanation)

        print("\n🎉 Advanced features demonstrated!")

    except Exception as e:
        print(f"❌ Advanced features error: {e}")


def main():
    """Run the complete query parsing demonstration."""

    print("🧠 LangStruct Query Parsing & Bidirectional RAG")
    print("=" * 50)

    # Ready to go!
    print("\n✅ Using LangStruct's unified API for both extraction and query parsing")

    try:
        # Run demonstrations
        demonstrate_basic_parsing()
        demonstrate_bidirectional_rag()
        demonstrate_advanced_features()

        print("\n" + "=" * 50)
        print("🎯 Key Takeaways:")
        print("   • Single LangStruct instance handles both extraction and queries")
        print("   • Natural language queries → Structured filters automatically")
        print("   • Same schema for both operations")
        print("   • Bidirectional RAG enhancement for precise retrieval")
        print("   • No more manual filter construction required")
        print("   • Makes structured RAG accessible to non-technical users")

        print("\n🚀 This completes the RAG enhancement circle!")
        print("   Traditional RAG: Documents + Semantic Search")
        print("   LangStruct RAG: Documents + Queries → Structured Intelligence")

    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\n💡 Common fixes:")
        print("   • Set your API key: export GOOGLE_API_KEY='your-key'")
        print("   • Ensure LangStruct is installed: pip install langstruct")
        print("   • Check network connectivity")


if __name__ == "__main__":
    main()
