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

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
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

        # IMPROVED Router prompt - prefer local documents when available
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query router for a RAG system.
Analyze the user's query and determine the best data source(s) to use.

Available sources:
- local_rag: For questions about content in UPLOADED DOCUMENTS/IMAGES
- web_search: For current events, real-time data, general knowledge NOT in documents
- hybrid: When query needs BOTH uploaded documents AND external web information

CRITICAL ROUTING RULES (when documents are uploaded):

1. **local_rag** (PREFERRED when documents uploaded):
   - Questions about specific things that could be in documents (sculptures, buildings, people, places, events, objects)
   - Queries about content, descriptions, or information that might be in uploaded files
   - Questions that don't require real-time/current data
   - Examples: "What is this sculpture?", "Describe the building", "What does the image show?"
   - Use local_rag FIRST to check if the answer is in uploaded content

2. **web_search** (ONLY for queries that CANNOT be in documents):
   - Current events, news, "latest", "recent", "today", real-time data
   - Weather, stock prices, live updates
   - General knowledge questions when NO documents are uploaded
   - Questions explicitly asking for external/current information

3. **hybrid** (When both needed):
   - "Compare my document with current industry standards"
   - "Update my document with latest research"
   - Queries that need both document content AND current web information

IMPORTANT ROUTING LOGIC:
- If user has uploaded documents/images → Try local_rag FIRST (unless clearly real-time query)
- If query is about specific things (sculptures, buildings, objects, people) → local_rag
- If query requires current/real-time data → web_search
- If query explicitly mentions documents → local_rag
- If NO documents uploaded → web_search

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

            # IMPROVED: When documents are uploaded, prefer local_rag for specific queries
            # Only override to web_search if it's clearly a real-time/current events query
            if has_uploaded_docs:
                query_lower = query.lower()
                real_time_keywords = [
                    'latest', 'current', 'recent', 'today', 'now', 'this week', 'this month',
                    'news', 'breaking', 'weather', 'temperature', 'forecast', 'stock', 'price',
                    'live', 'happening now', 'update', 'status'
                ]
                is_real_time_query = any(kw in query_lower for kw in real_time_keywords)
                
                # If it's NOT a real-time query and we have docs, prefer local_rag
                if not is_real_time_query and result.get("datasource") == "web_search":
                    # Check if query is about specific things that could be in documents
                    specific_thing_keywords = [
                        'sculpture', 'building', 'statue', 'monument', 'image', 'picture', 'photo',
                        'what is', 'what does', 'describe', 'show', 'about', 'where is', 'what is the'
                    ]
                    is_specific_query = any(kw in query_lower for kw in specific_thing_keywords)
                    
                    if is_specific_query:
                        logger.info(f"✅ Overriding web_search → local_rag (specific query with uploaded docs)")
                        result["datasource"] = "local_rag"
                        result["reasoning"] = "Query about specific content - checking uploaded documents first"
                        result["confidence"] = 0.8
                
                # Only override local_rag to web_search if it's clearly real-time
                elif result.get("datasource") == "local_rag" and is_real_time_query:
                    logger.info(f"⚠️ Real-time query detected, overriding local_rag → web_search")
                    result["datasource"] = "web_search"
                    result["reasoning"] = "Real-time/current events query - using web search"
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
            "history of", "background on",
            "什麼是", "什麼是", "誰", "解釋", "如何", "怎樣", "怎麼", 
            "定義", "介紹一下", "描述", "為什麼", "何時", "哪裡", 
            "天氣", "溫度", "預報", "新聞", "最新", "當前", "最近", 
            "歷史", "背景", "概述", "講講", "說明"
        ]

        # Document-specific indicators (NOT general knowledge)
        document_indicators = [
            "my document", "my file", "my pdf", "my paper",
            "the document", "the file", "the pdf", "the paper",
            "uploaded", "attachment",
            "according to my", "based on my",
            "in my file", "in the document",
            "我上傳", "我的文件", "我的文檔", "我的pdf", "我的論文", "剛剛上傳", 
            "上傳的", "本地文檔", "我的項目", "這個文檔", "這個文件", "上傳的項目",
            "根據我的", "基於我的", "在我的文件", "在文檔中", "文檔裡", "文件中",
            "這份文檔", "這個pdf", "這份文件", "我上傳的", "我剛剛上傳的"
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
            "what does my", "summarize my", "analyze my",
            "我上傳", "我的文件", "我的文檔", "我的pdf", "剛剛上傳", "上傳的",
            "本地文檔", "我的項目", "這個文檔", "這個文件", "上傳的項目",
            "根據我的文檔", "基於我的文件", "在我的文檔中", "文檔裡說",
            "這份文檔", "這個pdf", "這份文件", "我上傳的", "我剛剛上傳的",
            "介紹一下我上傳的", "總結我的文檔", "分析我的文件"
        ]

        # Web/external keywords
        web_keywords = [
            "latest", "current", "recent", "news", "today", "now",
            "what is", "what are", "explain", "define", "how does",
            "weather", "temperature", "forecast",
            "2025", "2024", "this year",
            "最新", "當前", "最近", "新聞", "今天", "現在",
            "什麼是", "解釋", "定義", "如何", "怎樣",
            "天氣", "溫度", "預報", "今年", "現在"
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

        # 3. Check if it's a real-time query (must use web_search)
        real_time_keywords = [
            'latest', 'current', 'recent', 'today', 'now', 'this week', 'this month',
            'last 5 days', 'last week', 'last month', 'yesterday',
            'news', 'breaking', 'weather', 'temperature', 'forecast',
            'stock', 'price', 'performance', 'trading', 'market', 'share price',
            'departure', 'arrival', 'schedule', 'timetable', 'bus', 'train', 'flight',
            'transit', 'transport', 'route', 'station', 'stop',
            'live', 'happening now', 'update', 'status', 'real-time', 'real time'
        ]
        is_real_time = any(kw in query_lower for kw in real_time_keywords)
        
        if is_real_time:
            return {
                "datasource": "web_search",
                "reasoning": "Real-time/current events query - using web search",
                "confidence": 0.9,
                "query": query
            }
        
        # 4. If has docs and query is about specific things → local_rag (preferred)
        specific_thing_keywords = [
            'sculpture', 'building', 'statue', 'monument', 'image', 'picture', 'photo',
            'what is', 'what does', 'describe', 'show', 'about', 'where is', 'what is the',
            'who is', 'what are', 'explain the'
        ]
        is_specific_query = any(kw in query_lower for kw in specific_thing_keywords)
        
        if has_uploaded_docs and is_specific_query:
            return {
                "datasource": "local_rag",
                "reasoning": "Query about specific content - checking uploaded documents first",
                "confidence": 0.8,
                "query": query
            }

        # 5. Has web intent → web_search
        if has_web_intent:
            return {
                "datasource": "web_search",
                "reasoning": "External information query - using web search",
                "confidence": 0.8,
                "query": query
            }

        # 6. DEFAULT: If docs available, try local_rag first; otherwise web_search
        if has_uploaded_docs:
            return {
                "datasource": "local_rag",
                "reasoning": "Default routing with uploaded documents - checking local content first",
                "confidence": 0.7,
                "query": query
            }
        else:
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
