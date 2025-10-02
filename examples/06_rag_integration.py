#!/usr/bin/env python3
"""
LangStruct + RAG Integration Example

This example demonstrates how to enhance RAG systems with structured metadata
extraction using LangStruct for superior filtering and retrieval capabilities.

Requirements:
    pip install langstruct[examples]  # Includes ChromaDB, LangChain, etc.
    # OR install individually:
    # pip install langstruct langchain-community langchain-text-splitters chromadb openai

Environment Variables:
    OPENAI_API_KEY - Required for OpenAI embeddings and LLM
    GOOGLE_API_KEY - Alternative LLM option (Gemini)

Usage:
    python 06_rag_integration.py
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Import required library
from langstruct import LangStruct

# Check for optional dependencies
try:
    from langchain.schema import Document
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Mock classes for demonstration when LangChain isn't available
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any]):
            self.page_content = page_content
            self.metadata = metadata


@dataclass
class RAGResult:
    """Container for RAG query results with metadata"""

    documents: List[Document]
    metadata_summary: Dict[str, Any]
    query: str
    filters: Optional[Dict[str, Any]] = None


class EnhancedRAGSystem:
    """RAG system enhanced with LangStruct metadata extraction"""

    def __init__(self, extraction_schema: Dict[str, Any]):
        """
        Initialize the enhanced RAG system

        Args:
            extraction_schema: Example schema for LangStruct metadata extraction
        """
        # Initialize LangStruct with error handling
        try:
            self.metadata_extractor = LangStruct(
                example=extraction_schema,
                # Model will use LangStruct's default unless specified
            )
            # Call self.metadata_extractor.optimize(...) later with labeled data if needed
        except Exception as e:
            raise ValueError(
                f"Failed to initialize LangStruct: {e}. "
                "Ensure you have OPENAI_API_KEY or GOOGLE_API_KEY set."
            )

        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.schema = extraction_schema

    def process_documents(self, documents: List[str]) -> List[Document]:
        """
        Process documents with LangStruct metadata extraction

        Args:
            documents: List of document texts to process

        Returns:
            List of enhanced documents with structured metadata
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", ".", "!", "?"]
        )

        enhanced_documents = []

        for i, doc_text in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}...")

            # Split document into chunks
            doc = Document(page_content=doc_text, metadata={"doc_id": i})
            chunks = text_splitter.split_documents([doc])

            for chunk_idx, chunk in enumerate(chunks):
                try:
                    # Extract structured metadata using LangStruct
                    extraction = self.metadata_extractor.extract(chunk.page_content)

                    # Prepare metadata
                    metadata = {
                        "doc_id": i,
                        "chunk_id": chunk_idx,
                        "langstruct_confidence": extraction.confidence,
                        "extraction_sources": extraction.sources,  # Sources always available
                        **extraction.entities,  # Add extracted structured data
                    }

                    # Create enhanced document
                    enhanced_doc = Document(
                        page_content=chunk.page_content, metadata=metadata
                    )
                    enhanced_documents.append(enhanced_doc)

                except Exception as e:
                    print(f"Error processing chunk {chunk_idx} of document {i}: {e}")
                    # Add document without enhanced metadata on error
                    chunk.metadata.update({"doc_id": i, "chunk_id": chunk_idx})
                    enhanced_documents.append(chunk)

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=enhanced_documents,
            embedding=self.embeddings,
            collection_name="langstruct_enhanced_rag",
        )

        print(
            f"Created vector store with {len(enhanced_documents)} enhanced document chunks"
        )
        return enhanced_documents

    def query(
        self,
        query_text: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
    ) -> RAGResult:
        """
        Query the enhanced RAG system with optional metadata filtering

        Args:
            query_text: Natural language query
            metadata_filters: Optional metadata filters for precise retrieval
            k: Number of results to return

        Returns:
            RAGResult with documents and metadata summary
        """
        if not self.vectorstore:
            raise ValueError("No documents processed. Call process_documents() first.")

        # Convert filters to Chroma format
        where_clause = self._build_where_clause(metadata_filters)

        # Perform similarity search with metadata filtering
        results = self.vectorstore.similarity_search(
            query=query_text, k=k, where=where_clause if where_clause else None
        )

        # Generate metadata summary
        metadata_summary = self._summarize_metadata(results)

        return RAGResult(
            documents=results,
            metadata_summary=metadata_summary,
            query=query_text,
            filters=metadata_filters,
        )

    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict]:
        """Convert filters to Chroma where clause format"""
        if not filters:
            return None

        where_clause = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle range queries ($gte, $lt, etc.)
                for op, val in value.items():
                    if op in ["$gte", "$gt", "$lt", "$lte", "$eq", "$ne"]:
                        where_clause[key] = {op: val}
                    elif op == "$in":
                        where_clause[key] = {"$in": val}
            elif isinstance(value, list):
                # List values as $in query
                where_clause[key] = {"$in": value}
            else:
                # Exact match
                where_clause[key] = {"$eq": value}

        return where_clause

    def _summarize_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate summary statistics from retrieved documents"""
        summary = {
            "total_results": len(documents),
            "confidence_scores": [],
            "metadata_fields": set(),
        }

        for doc in documents:
            if "langstruct_confidence" in doc.metadata:
                summary["confidence_scores"].append(
                    doc.metadata["langstruct_confidence"]
                )
            summary["metadata_fields"].update(doc.metadata.keys())

        # Calculate confidence statistics
        if summary["confidence_scores"]:
            summary["avg_confidence"] = sum(summary["confidence_scores"]) / len(
                summary["confidence_scores"]
            )
            summary["min_confidence"] = min(summary["confidence_scores"])
            summary["max_confidence"] = max(summary["confidence_scores"])

        summary["metadata_fields"] = list(summary["metadata_fields"])

        return summary


def demo_financial_rag():
    """Demonstrate enhanced RAG with financial documents"""

    print("=== Financial Document RAG Demo ===\n")

    # Define financial metadata schema
    financial_schema = {
        "company_name": "Apple Inc.",
        "quarter": "Q3 2024",
        "revenue": "125.3 billion",
        "revenue_numeric": 125.3,
        "profit_margin": "23.1%",
        "profit_margin_numeric": 23.1,
        "key_metrics": ["iPhone sales", "Services growth"],
        "risks": ["Supply chain", "Regulatory"],
        "document_type": "10-Q",
        "fiscal_year": 2024,
        "growth_rate": 12.5,
    }

    # Sample financial documents (in practice, load from files)
    sample_documents = [
        """
        Apple Inc. Q3 2024 Earnings Report

        Apple reported record Q3 2024 revenue of $85.4 billion, up 15.2% from the prior year.
        The company's profit margin improved to 23.1%, driven by strong iPhone sales and
        continued growth in Services. Key highlights include iPhone revenue of $45.8 billion
        and Services revenue of $22.3 billion.

        Looking forward, Apple faces several risks including supply chain constraints in Asia
        and potential regulatory changes affecting App Store operations. The company continues
        to invest heavily in AI and machine learning capabilities.
        """,
        """
        Microsoft Corporation Q3 2024 Financial Results

        Microsoft delivered strong Q3 2024 performance with revenue of $61.9 billion,
        representing 17% year-over-year growth. The company's profit margin reached 19.8%,
        benefiting from Azure cloud services expansion and productivity software growth.

        Azure revenue grew 31% in the quarter, while Microsoft 365 commercial products
        saw 15% growth. Key risk factors include increased competition in cloud services
        and potential cybersecurity challenges. The company's AI initiatives, including
        Copilot integration, are driving significant customer adoption.
        """,
        """
        Alphabet Inc. Q3 2024 Earnings

        Alphabet reported Q3 2024 revenue of $76.7 billion, up 11% year-over-year, with
        a profit margin of 21.3%. Google Search revenue totaled $44.9 billion while
        Google Cloud achieved $8.4 billion in revenue, up 35% from the prior year.

        The company faces regulatory risks in multiple jurisdictions and increased
        competition in AI search. YouTube advertising revenue reached $7.9 billion.
        Alphabet continues significant investments in AI research and infrastructure,
        with particular focus on LLM development and deployment.
        """,
    ]

    # Initialize enhanced RAG system
    rag_system = EnhancedRAGSystem(financial_schema)

    # Process documents with metadata extraction
    enhanced_docs = rag_system.process_documents(sample_documents)

    print(f"Processed {len(enhanced_docs)} document chunks with structured metadata\n")

    # Demonstrate various query types
    demo_queries = [
        {
            "name": "High Revenue Companies",
            "query": "Q3 2024 financial performance",
            "filters": {"revenue_numeric": {"$gte": 60.0}},
            "description": "Find companies with Q3 revenue >= $60B",
        },
        {
            "name": "High Growth Technology",
            "query": "cloud services and AI technology growth",
            "filters": {
                "key_metrics": ["Azure", "AI", "cloud"],
                "growth_rate": {"$gte": 15.0},
            },
            "description": "Tech companies with strong AI/cloud growth",
        },
        {
            "name": "Regulatory Risk Analysis",
            "query": "regulatory challenges and compliance",
            "filters": {"risks": ["Regulatory"]},
            "description": "Companies facing regulatory risks",
        },
        {
            "name": "Profitable Tech Giants",
            "query": "technology company profitability analysis",
            "filters": {
                "profit_margin_numeric": {"$gte": 20.0},
                "revenue_numeric": {"$gte": 70.0},
            },
            "description": "Large profitable tech companies",
        },
    ]

    # Execute demo queries
    for query_demo in demo_queries:
        print(f"=== {query_demo['name']} ===")
        print(f"Description: {query_demo['description']}")
        print(f"Query: '{query_demo['query']}'")
        print(f"Filters: {json.dumps(query_demo['filters'], indent=2)}")

        # Execute query
        result = rag_system.query(
            query_demo["query"], metadata_filters=query_demo["filters"], k=3
        )

        print(f"Found {result.metadata_summary['total_results']} results")
        if result.metadata_summary.get("avg_confidence"):
            print(
                f"Average confidence: {result.metadata_summary['avg_confidence']:.2f}"
            )

        # Show results
        for i, doc in enumerate(result.documents, 1):
            print(f"\n  Result {i}:")
            print(f"    Company: {doc.metadata.get('company_name', 'Unknown')}")
            print(f"    Revenue: ${doc.metadata.get('revenue_numeric', 'N/A')}B")
            print(
                f"    Profit Margin: {doc.metadata.get('profit_margin_numeric', 'N/A')}%"
            )
            print(f"    Key Metrics: {doc.metadata.get('key_metrics', [])}")
            print(
                f"    Confidence: {doc.metadata.get('langstruct_confidence', 'N/A'):.2f}"
            )
            print(f"    Preview: {doc.page_content[:150]}...")

        print("\n" + "=" * 80 + "\n")


def demo_comparison():
    """Show traditional RAG vs LangStruct-enhanced RAG comparison"""

    print("=== Traditional RAG vs LangStruct-Enhanced RAG ===\n")

    print(
        "Example Query: 'Find Q3 2024 tech companies with revenue over $60B discussing AI'\n"
    )

    print("Traditional RAG Approach:")
    print("  • Searches for: 'Q3' OR '2024' OR 'revenue' OR 'AI'")
    print("  • Returns: 50+ documents (many irrelevant)")
    print("  • Results include:")
    print("    - Q1 2023 reports (wrong quarter/year)")
    print("    - Companies with $10M revenue (too small)")
    print("    - Non-tech companies mentioning AI briefly")
    print("  • User must manually filter through all results")

    print("\nLangStruct-Enhanced Approach:")
    print("  • Extracts requirements: quarter='Q3 2024', revenue>60B, mentions AI")
    print("  • Returns: 3-5 highly relevant documents")
    print("  • Results guaranteed to have:")
    print("    ✓ Correct quarter (Q3 2024)")
    print("    ✓ Revenue exceeding $60B")
    print("    ✓ Substantial AI discussion")
    print("  • User gets exactly what they need immediately")

    print("\n" + "=" * 60)
    print("\nKey Benefits:")
    print("  • Precision: Get exactly what you ask for")
    print("  • Efficiency: No manual filtering needed")
    print("  • Scalability: Works with millions of documents")
    print("  • Domain-agnostic: Adapts to any field")


def main():
    """Main demo function"""
    print("LangStruct + RAG Integration Demo")
    print("=" * 50)

    # Check for optional dependencies
    if not LANGCHAIN_AVAILABLE:
        print("WARNING: LangChain dependencies not found.")
        print(
            "Install with: pip install langchain-community langchain-text-splitters chromadb"
        )
        print("\nProceeding with demonstration code only...\n")

        # Show comparison without actually running queries
        demo_comparison()
        return

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment variables.")
        print("Set your OpenAI API key to run the full demo:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nProceeding with demonstration code only...\n")

        # Show comparison without actually running queries
        demo_comparison()
        return

    try:
        # Run full demo with actual embeddings and queries
        demo_financial_rag()
        demo_comparison()

        print("\n=== Next Steps ===")
        print("1. Try with your own documents:")
        print("   - Replace sample_documents with your file contents")
        print("   - Adjust the schema for your domain")
        print("   - Experiment with different query filters")
        print("\n2. Explore other domains:")
        print("   - Medical records: patient demographics, conditions, treatments")
        print("   - Legal contracts: parties, terms, risks, jurisdictions")
        print("   - Research papers: authors, methodologies, findings")
        print("\n3. Production considerations:")
        print("   - Implement batch processing for large document sets")
        print("   - Add confidence threshold filtering")
        print("   - Set up monitoring for extraction quality")

    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure you have installed all required packages:")
        print(
            "pip install langstruct langchain-community langchain-text-splitters chromadb openai"
        )


if __name__ == "__main__":
    main()
