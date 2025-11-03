"""
Query Classifier Module 
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Intelligent query classifier with improved routing logic
    Determines whether to use local RAG, web search, or hybrid approach
    based on query content and uploaded document availability.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize the query classifier

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=self.api_key
        )

        # IMPROVED Router prompt - clearer decision criteria
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query router for a RAG system.
Analyze the user's query and determine the best data source(s) to use.

Available sources:
- local_rag: ONLY for questions EXPLICITLY about UPLOADED DOCUMENTS
- web_search: For ALL general knowledge, facts, definitions, current events, external information
- hybrid: ONLY when the query explicitly needs BOTH uploaded documents AND external web information

CRITICAL ROUTING RULES:

1. **web_search** (DEFAULT for most queries):
   - General knowledge questions (e.g., "What is X?", "Explain Y", "How does Z work?")
   - Current events, news, "latest", "recent", real-time data
   - Facts, definitions, explanations NOT about uploaded files
   - Historical information, science, technology
   - ANY question that doesn't explicitly mention documents

2. **local_rag** (ONLY when EXPLICITLY about documents):
   - Query mentions: "my document", "uploaded file", "the PDF", "in my paper"
   - Asks about: "what does my document say", "according to my file"
   - References specific uploaded content

3. **hybrid** (Rare - only when BOTH needed):
   - "Compare my document with current industry standards"
   - "Update my document with latest research"
   - Explicitly asks to combine document content with web information

IMPORTANT:
- If query is general knowledge → web_search (even if user has uploaded docs)
- If query is about weather, news, definitions → web_search
- If query doesn't mention documents → web_search
- Only use local_rag if query EXPLICITLY references uploaded content

Respond with ONLY a JSON object:
{{
  "datasource": "local_rag" or "web_search" or "hybrid",
  "reasoning": "brief explanation",
  "confidence": 0.85
}}

No other text, just the JSON."""),
            ("human", "User Query: {query}\n\nContext: {context_info}\n\nRoute this query:")
        ])

        # Create chain
        self.router_chain = self.router_prompt | self.llm
        logger.info("✅ Query Classifier initialized with improved routing")

    def classify_query(self, query: str, has_uploaded_docs: bool = False) -> Dict[str, Any]:
        """
        Classify a query and determine routing
        IMPROVED: Better logic for general knowledge questions

        Args:
            query: The user's query string
            has_uploaded_docs: Whether the user has uploaded any documents

        Returns:
            Dictionary with routing decision and metadata
        """
        # Build context information
        context_info = f"User has {'uploaded documents available' if has_uploaded_docs else 'NO uploaded documents'}"
        
        try:
            # Get routing decision from LLM
            response = self.router_chain.invoke({
                "query": query,
                "context_info": context_info
            })

            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)

            # Parse JSON from response
            result = self._parse_json_response(content)

            # Validate datasource
            valid_sources = ["local_rag", "web_search", "hybrid"]
            if result.get("datasource") not in valid_sources:
                result["datasource"] = "web_search"

            # CRITICAL FIX: If no docs uploaded, always use web_search
            if not has_uploaded_docs and result.get("datasource") in ["local_rag", "hybrid"]:
                logger.info(f"⚠️ No docs available, overriding {result['datasource']} → web_search")
                result["datasource"] = "web_search"
                result["reasoning"] = "No local documents available - using web search"
                result["confidence"] = 0.9

            # Additional validation: Check if it's a general knowledge question
            if has_uploaded_docs and result.get("datasource") == "local_rag":
                if self._is_general_knowledge(query):
                    logger.info(f"⚠️ General knowledge question detected, overriding local_rag → web_search")
                    result["datasource"] = "web_search"
                    result["reasoning"] = "General knowledge question - using web search"
                    result["confidence"] = 0.85

            logger.info(f"✅ Classified '{query[:50]}...' → {result['datasource']}")

            return {
                "datasource": result.get("datasource", "web_search"),
                "reasoning": result.get("reasoning", "Default routing"),
                "confidence": float(result.get("confidence", 0.7)),
                "query": query
            }

        except Exception as e:
            logger.warning(f"⚠️ Classification error: {e}. Using fallback logic.")
            return self._fallback_classification(query, has_uploaded_docs)

    def _is_general_knowledge(self, query: str) -> bool:
        """
        Check if query is a general knowledge question
        (should use web_search even if docs are available)
        """
        query_lower = query.lower()

        # General knowledge indicators
        general_knowledge_patterns = [
            "what is", "what are", "who is", "who are",
            "explain", "how does", "how do", "how to",
            "define", "definition of",
            "tell me about", "describe",
            "why", "when", "where",
            "weather", "temperature", "forecast",
            "news", "latest", "current", "recent",
            "history of", "background on"
        ]

        # Document-specific indicators (NOT general knowledge)
        document_indicators = [
            "my document", "my file", "my pdf", "my paper",
            "the document", "the file", "the pdf", "the paper",
            "uploaded", "attachment",
            "according to my", "based on my",
            "in my file", "in the document"
        ]

        # Check if it has document indicators (if yes, not general knowledge)
        has_doc_indicator = any(indicator in query_lower for indicator in document_indicators)
        if has_doc_indicator:
            return False

        # Check if it has general knowledge patterns
        has_general_pattern = any(pattern in query_lower for pattern in general_knowledge_patterns)
        if has_general_pattern:
            return True

        # Default: if it doesn't mention documents explicitly, treat as general knowledge
        return True

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling various formats"""
        try:
            # Try direct JSON parse
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass

            # Manual extraction as last resort
            result = {}

            # Extract datasource
            if 'local_rag' in content.lower():
                result['datasource'] = 'local_rag'
            elif 'hybrid' in content.lower():
                result['datasource'] = 'hybrid'
            else:
                result['datasource'] = 'web_search'

            # Extract reasoning
            reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', content)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1)
            else:
                result['reasoning'] = "Classification based on query content"

            # Extract confidence
            confidence_match = re.search(r'"confidence":\s*([0-9.]+)', content)
            if confidence_match:
                result['confidence'] = float(confidence_match.group(1))
            else:
                result['confidence'] = 0.7

            return result

    def _fallback_classification(self, query: str, has_uploaded_docs: bool) -> Dict[str, Any]:
        """
        Improved fallback classification
        FIXED: Defaults to web_search for general knowledge
        """
        query_lower = query.lower()

        # EXPLICIT document keywords (must be very specific)
        document_keywords = [
            "my document", "my file", "my pdf", "my paper",
            "the document", "the file", "the pdf", "uploaded file",
            "in my document", "according to my document",
            "what does my", "summarize my", "analyze my"
        ]

        # Web/external keywords
        web_keywords = [
            "latest", "current", "recent", "news", "today", "now",
            "what is", "what are", "explain", "define", "how does",
            "weather", "temperature", "forecast",
            "2025", "2024", "this year"
        ]

        # Check for EXPLICIT document references
        has_explicit_doc_ref = any(kw in query_lower for kw in document_keywords)

        # Check for web intent
        has_web_intent = any(kw in query_lower for kw in web_keywords)

        # DECISION LOGIC (improved):

        # 1. If NO docs uploaded → always web_search
        if not has_uploaded_docs:
            return {
                "datasource": "web_search",
                "reasoning": "No local documents available - using web search",
                "confidence": 0.9,
                "query": query
            }

        # 2. If EXPLICIT document reference AND has docs → local_rag
        if has_explicit_doc_ref and has_uploaded_docs:
            if has_web_intent:
                # Both document and web intent → hybrid
                return {
                    "datasource": "hybrid",
                    "reasoning": "Query mentions both uploaded documents and external information",
                    "confidence": 0.75,
                    "query": query
                }
            else:
                # Only document intent → local_rag
                return {
                    "datasource": "local_rag",
                    "reasoning": "Query explicitly references uploaded documents",
                    "confidence": 0.85,
                    "query": query
                }

        # 3. Has web intent OR general knowledge → web_search
        if has_web_intent or self._is_general_knowledge(query):
            return {
                "datasource": "web_search",
                "reasoning": "General knowledge or external information query",
                "confidence": 0.8,
                "query": query
            }

        # 4. DEFAULT: web_search (not local_rag!)
        # This is the key fix - default should be web_search
        return {
            "datasource": "web_search",
            "reasoning": "General query - using web search by default",
            "confidence": 0.7,
            "query": query
        }


# Test function
if __name__ == "__main__":
    print("=" * 80)
    print("IMPROVED QUERY CLASSIFIER TEST")
    print("=" * 80)

    try:
        classifier = QueryClassifier()

        # Test queries with expected routing
        test_queries = [
            # General knowledge (should be web_search even with docs)
            ("What is machine learning?", True, "web_search"),
            ("Explain neural networks", True, "web_search"),
            ("How does AI work?", True, "web_search"),
            ("What's the weather in Hong Kong?", True, "web_search"),
            ("Latest AI news", True, "web_search"),
            
            # Explicit document queries (should be local_rag)
            ("What does my document say about AI?", True, "local_rag"),
            ("Summarize my uploaded PDF", True, "local_rag"),
            ("According to my document, what is the conclusion?", True, "local_rag"),
            
            # Hybrid queries
            ("Compare my document with current industry trends", True, "hybrid"),
            
            # No documents (always web_search)
            ("What is Python?", False, "web_search"),
        ]

        correct = 0
        total = len(test_queries)

        for query, has_docs, expected in test_queries:
            result = classifier.classify_query(query, has_uploaded_docs=has_docs)
            actual = result['datasource']
            is_correct = actual == expected

            status = "✅" if is_correct else "❌"
            if is_correct:
                correct += 1

            print(f"\n{status} Query: {query}")
            print(f"   Has Docs: {has_docs}")
            print(f"   Expected: {expected}")
            print(f"   Actual: {actual}")
            print(f"   Reasoning: {result['reasoning']}")
            print(f"   Confidence: {result['confidence']:.2f}")

        print("\n" + "=" * 80)
        print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
        print("=" * 80)

        if correct == total:
            print("✅ All tests passed!")
        else:
            print(f"⚠️ {total - correct} test(s) failed")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
