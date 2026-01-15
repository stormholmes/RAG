import asyncio
from typing import Dict, Any, List, Optional
from query_classifier import QueryClassifier
from web_search_tavily import WebSearchEnhanced, HybridSearchEnhanced
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentSourceRouter:
    """
    Async Enhanced Intelligent Router with Tavily Integration
    Optimized for low latency using asyncio and parallel execution.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_web_search: bool = True,
        web_max_results: int = 5,
        fetch_full_content: bool = True
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.enable_web_search = enable_web_search
        self.web_max_results = web_max_results
        self.fetch_full_content = fetch_full_content
        
        # Initialize components
        try:
            self.query_classifier = QueryClassifier(api_key=self.api_key)
            logger.info("✅ Query Classifier initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Query Classifier: {e}")
            self.query_classifier = None
        
        if enable_web_search:
            try:
                self.web_search = WebSearchEnhanced(max_results=web_max_results)
                # HybridSearchEnhanced is not strictly needed if we handle hybrid logic here manually
                self.hybrid_search = HybridSearchEnhanced(self.web_search)
                logger.info("✅ Enhanced Web Search (Tavily) initialized")
            except Exception as e:
                logger.warning(f"⚠️ Web search initialization failed: {e}")
                self.web_search = None
        else:
            self.web_search = None
        
        self.routing_history = []
        
        # Optimization: Pre-load reranker model if possible to avoid cold-start latency
        self.cross_encoder = None
        # self._preload_reranker() # Uncomment to load on startup (increases startup time, decreases first request time)

    async def route_query(
        self,
        query: str,
        local_retriever=None,
        has_uploaded_docs: bool = False
    ) -> Dict[str, Any]:
        """
        Async Route a query with Tavily as primary backend.
        Runs classification and retrieval steps.
        """
        logger.info(f"🔄 [Async] Routing query: {query[:50]}...")
        
        result = {
            'query': query,
            'routing': {
                'datasource': 'web_search',
                'reasoning': 'Default routing',
                'confidence': 0.5
            },
            'context': '',
            'sources': [],
            'raw_search_results': []
        }
        
        try:
            # Step 1: Classify query (Run in thread to avoid blocking loop if classifier is sync)
            datasource = 'web_search' # Default
            
            if self.query_classifier is None:
                logger.warning("⚠️ Query classifier not available, using fallback")
                datasource = self._fallback_datasource_selection(has_uploaded_docs, query)
                result['routing']['datasource'] = datasource
            else:
                # Wrap synchronous classifier in thread
                classification = await asyncio.to_thread(
                    self.query_classifier.classify_query, 
                    query, 
                    has_uploaded_docs
                )
                result['routing'] = classification
                datasource = result['routing']['datasource']
            
            logger.info(f"📍 Selected datasource: {datasource}")
            
            # Step 2: Execute retrieval based on classification
            retrieval_result = {}
            
            if datasource == 'local_rag':
                retrieval_result = await self._retrieve_local(query, local_retriever)
            elif datasource == 'web_search':
                retrieval_result = await self._retrieve_web(query)
            elif datasource == 'hybrid':
                retrieval_result = await self._retrieve_hybrid(query, local_retriever)
            else:
                logger.warning(f"Unknown datasource: {datasource}, defaulting to web_search")
                retrieval_result = await self._retrieve_web(query)

            result.update(retrieval_result)

            # Step 3: Save routing history (Non-blocking)
            self._save_routing_history(result)
            
            logger.info(f"✅ Query routed successfully to {datasource}")
            
        except Exception as e:
            logger.error(f"❌ Routing error: {e}", exc_info=True)
            result['error'] = str(e)
            result['context'] = f"Error during routing: {str(e)}"
        
        return result

    async def _retrieve_web(self, query: str) -> Dict[str, Any]:
        """
        Parallelized Web Retrieval
        executes main search, historical search, and targeted search concurrently.
        """
        logger.info("🌐 Performing parallel web search...")
        
        if self.web_search is None:
            return {'context': 'Web search disabled.', 'sources': [], 'retrieval_type': 'web_search'}
        
        try:
            # 1. Identify all needed search queries first
            tasks = []
            
            # Task A: Main Query
            tasks.append(self._execute_single_search(query))
            
            # Task B: Historical Queries (Limit 2)
            historical_searches = self._identify_historical_searches(query)
            for term in historical_searches[:2]:
                tasks.append(self._execute_single_search(term))
                
            # Task C: Targeted Queries
            targeted_searches = self._identify_targeted_searches(query)
            for term in targeted_searches:
                tasks.append(self._execute_single_search(term))

            # Task D: Multi-part (Only if needed, but for latency, we might skip dependent logic 
            # or launch them all if we are aggressive. Here we skip complex multi-part dependency 
            # to save time, or we could run them in parallel if independent)
            
            # 2. Run all searches in parallel
            logger.info(f"🚀 Launching {len(tasks)} concurrent search tasks")
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 3. Aggregation & Deduplication
            web_results = []
            seen_urls = set()
            
            for res_list in results_lists:
                if isinstance(res_list, Exception):
                    logger.error(f"Search task failed: {res_list}")
                    continue
                if not res_list:
                    continue
                    
                for item in res_list:
                    url = item.get('url', '')
                    if url not in seen_urls:
                        web_results.append(item)
                        seen_urls.add(url)
            
            # Limit total results
            web_results = web_results[:10]
            
            if not web_results:
                return {'context': 'No results.', 'sources': [], 'num_results': 0}
            
            # 4. Format Output
            return self._format_web_results(query, web_results)
            
        except Exception as e:
            logger.error(f"❌ Web search error: {e}", exc_info=True)
            return {'error': str(e), 'context': str(e), 'sources': []}

    async def _execute_single_search(self, query: str) -> List[Dict]:
        """Helper to run a synchronous Tavily search in a thread"""
        try:
            # Run the blocking search_and_extract in a separate thread
            return await asyncio.to_thread(
                self.web_search.search_and_extract,
                query,
                fetch_content=self.fetch_full_content
            )
        except Exception as e:
            logger.error(f"Single search failed for '{query}': {e}")
            return []

    async def _retrieve_local(self, query: str, local_retriever) -> Dict[str, Any]:
        """Async wrapper for local retrieval and reranking"""
        if not local_retriever:
            return {'context': 'No retriever.', 'sources': [], 'retrieval_type': 'local_rag'}

        try:
            # Run vector search in thread
            docs = await asyncio.to_thread(local_retriever.get_relevant_documents, query)
            
            if not docs:
                return {'context': 'No local docs.', 'sources': [], 'num_results': 0}

            # Run Reranking in thread (CPU bound)
            if len(docs) > 1:
                docs = await asyncio.to_thread(self._rerank_documents, query, docs, 4)

            # Formatting (Fast enough to run in loop)
            context = "=== 📄 LOCAL DOCUMENT RESULTS ===\n\n"
            sources = []
            for i, doc in enumerate(docs, 1):
                # ... (Keep existing formatting logic)
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content
                context += f"[Document {i}] {source}\nContent: {content}\n\n"
                sources.append({'type': 'local', 'source': source, 'content': content})

            return {
                'context': context, 
                'sources': sources, 
                'retrieval_type': 'local_rag',
                'num_results': len(docs)
            }
        except Exception as e:
            logger.error(f"Local retrieval error: {e}")
            return {'error': str(e), 'sources': []}

    async def _retrieve_hybrid(self, query: str, local_retriever) -> Dict[str, Any]:
        """
        True Parallel Hybrid Retrieval
        """
        logger.info("🔄 Performing PARALLEL hybrid retrieval...")
        
        # Launch both retrievals simultaneously
        task_local = self._retrieve_local(query, local_retriever)
        task_web = self._retrieve_web(query)
        
        # Wait for both
        local_res, web_res = await asyncio.gather(task_local, task_web)
        
        # Combine Results
        sources = local_res.get('sources', []) + web_res.get('sources', [])
        
        # Combine Context strings
        hybrid_context = f"=== 🔄 HYBRID RETRIEVAL RESULTS ===\n\n"
        hybrid_context += local_res.get('context', '') + "\n"
        hybrid_context += web_res.get('context', '')
        
        return {
            'context': hybrid_context,
            'sources': sources,
            'retrieval_type': 'hybrid',
            'num_local': local_res.get('num_results', 0),
            'num_web': web_res.get('num_results', 0)
        }

    # --- Existing Helper Methods (Keep mostly same, ensure they are thread-safe) ---

    def _format_web_results(self, query, results):
        """Helper to format web results (extracted from original _retrieve_web)"""
        context = f"=== 🌐 WEB SEARCH RESULTS (Tavily Enhanced) ===\nQuery: {query}\n\n"
        sources = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Unknown')
            content = result.get('full_content') or result.get('snippet', '')
            url = result.get('url', '')
            
            context += f"[Result {i}] {title}\nURL: {url}\nContent:\n{content}\n{'-'*50}\n"
            sources.append({
                'type': 'web',
                'title': title,
                'url': url,
                'full_content': content,
                'source': result.get('source', 'Web')
            })
            
        return {
            'context': context,
            'sources': sources,
            'retrieval_type': 'web_search',
            'num_results': len(results),
            'raw_search_results': results
        }

    def _rerank_documents(self, query: str, documents: List, top_k: int = 4) -> List:
        """
        Standard Reranking logic.
        WARNING: CrossEncoder is heavy. We run this in a thread via _retrieve_local.
        """
        if len(documents) <= 1:
            return documents
        
        try:
            # Lazy load inside thread to avoid global blocking, 
            # or better: load in __init__ if memory permits.
            from sentence_transformers import CrossEncoder
            
            if not hasattr(self, 'cross_encoder') or self.cross_encoder is None:
                # Use a lighter model for speed vs quality trade-off if needed
                # 'cross-encoder/ms-marco-TinyBERT-L-2-v2' is much faster than MiniLM
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.cross_encoder.predict(pairs)
            
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            return [doc for score, doc in scored_docs[:top_k]]
            
        except Exception as e:
            logger.warning(f"Reranking fallback to BM25/Original: {e}")
            return documents[:top_k]

    # Keep original helper methods: 
    # _identify_historical_searches, _identify_targeted_searches, 
    # _detect_multi_part_query, _fallback_datasource_selection, _save_routing_history
    # ... (Copy these verbatim from your source file as they are pure logic, fast enough) ...

    def _identify_historical_searches(self, query: str) -> List[str]:
        """
        Identify if query asks for historical/multi-year data and generate enhanced search queries
        
        Examples:
        - "past ten years" → add "historical", "year by year", "timeline"
        - "over the past 5 years" → add "2019-2024", "historical data"
        
        Returns:
            List of enhanced search queries, or empty list
        """
        query_lower = query.lower()
        historical_searches = []
        
        # Check for historical query patterns
        historical_patterns = [
            'past ten years', 'past 10 years', 'last ten years', 'last 10 years',
            'past five years', 'past 5 years', 'last five years', 'last 5 years',
            'past.*years', 'over the past', 'historical', 'year by year', 'timeline'
        ]
        
        import re
        is_historical = any(re.search(pattern, query_lower) for pattern in historical_patterns)
        
        if is_historical:
            # Add enhanced search terms to get more comprehensive historical data
            enhanced_queries = [
                query + " historical data",
                query + " year by year",
                query + " timeline",
                query + " all years"
            ]
            historical_searches.extend(enhanced_queries)
            logger.info(f"🔍 Historical query detected, adding enhanced searches: {enhanced_queries[:2]}")
        
        return historical_searches

    def _identify_targeted_searches(self, query: str) -> List[str]:
        """
        Identify if query mentions specific information that needs targeted search
        
        Examples:
        - "Douban score" → search for "Douban score [movie name]" or "豆瓣评分 [movie name]"
        - "IMDB rating" → search for "IMDB rating [movie name]"
        
        Returns:
            List of targeted search terms, or empty list
        """
        query_lower = query.lower()
        targeted_searches = []
        
        # Check for Douban score mentions
        if 'douban' in query_lower or '豆瓣' in query:
            # Try to extract movie name from context
            # Look for patterns like "movie they won for", "film", "movie", etc.
            import re
            
            # Try to find movie name mentioned in query
            movie_patterns = [
                r'movie they (?:won|received|got) (?:for|with)',
                r'film they (?:won|received|got) (?:for|with)',
                r'movie "([^"]+)"',
                r'film "([^"]+)"',
                r'movie ([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
            ]
            
            movie_name = None
            for pattern in movie_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    movie_name = match.group(1) if match.groups() else None
                    if movie_name:
                        break
            
            if movie_name:
                targeted_searches.append(f"Douban score {movie_name}")
                targeted_searches.append(f"豆瓣评分 {movie_name}")
            else:
                # If movie name not found, search for Douban score in general context
                targeted_searches.append("Douban score")
        
        # Check for IMDB rating mentions
        if 'imdb' in query_lower or 'imdb rating' in query_lower:
            # Similar logic for IMDB
            targeted_searches.append("IMDB rating")
        
        return targeted_searches

    def _fallback_datasource_selection(self, has_uploaded_docs: bool, query: str) -> str:
        """Fallback keyword-based routing"""
        query_lower = query.lower()
        
        web_keywords = [
            'latest', 'current', 'recent', 'news', 'today', 'now', 'update',
            '2024', '2025', 'trending', 'new', 'latest news', 'current events',
            'breaking', 'recent developments', 'today news'
        ]
        
        local_keywords = [
            'document', 'file', 'uploaded', 'pdf', 'image', 'my',
            'attachment', 'what does', 'according to', 'based on'
        ]
        
        has_web_intent = any(kw in query_lower for kw in web_keywords)
        has_local_intent = any(kw in query_lower for kw in local_keywords)
        
        if has_web_intent and has_local_intent and has_uploaded_docs:
            return 'hybrid'
        elif has_local_intent and has_uploaded_docs:
            return 'local_rag'
        else:
            return 'web_search'

    def _save_routing_history(self, result: Dict[str, Any]):
        """Save routing decision to history"""
        try:
            history_entry = {
                'query': result.get('query', ''),
                'datasource': result.get('routing', {}).get('datasource', 'unknown'),
                'reasoning': result.get('routing', {}).get('reasoning', ''),
                'confidence': result.get('routing', {}).get('confidence', 0),
                'retrieval_type': result.get('retrieval_type', 'unknown'),
                'num_sources': len(result.get('sources', [])),
                'error': result.get('error', None),
                'timestamp': datetime.now().isoformat()
            }
            
            self.routing_history.append(history_entry)
            logger.debug(f"📊 Routing history saved: {history_entry['datasource']}")
        except Exception as e:
            logger.error(f"Error saving routing history: {e}")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions"""
        if not self.routing_history:
            return {
                'total_queries': 0,
                'by_source': {},
                'by_retrieval_type': {},
                'avg_confidence': 0.0,
                'error_count': 0
            }
        
        total = len(self.routing_history)
        by_source = {}
        by_retrieval_type = {}
        total_confidence = 0.0
        error_count = 0
        
        for entry in self.routing_history:
            source = entry['datasource']
            by_source[source] = by_source.get(source, 0) + 1
            
            ret_type = entry.get('retrieval_type', 'unknown')
            by_retrieval_type[ret_type] = by_retrieval_type.get(ret_type, 0) + 1
            
            total_confidence += entry['confidence']
            
            if entry.get('error'):
                error_count += 1
        
        return {
            'total_queries': total,
            'by_source': by_source,
            'by_retrieval_type': by_retrieval_type,
            'avg_confidence': total_confidence / total if total > 0 else 0.0,
            'error_count': error_count,
            'success_rate': (total - error_count) / total if total > 0 else 0.0,
            'routing_history': self.routing_history
        }


# ==================== ASYNC TEST RUNNER ====================
if __name__ == "__main__":
    async def main():
        print("=" * 80)
        print("ASYNC INTELLIGENT ROUTER TEST")
        print("=" * 80)
        
        router = IntelligentSourceRouter(
            enable_web_search=True,
            web_max_results=3
        )
        
        # Example Query
        query = "What are the latest AI developments in 2025 and 2024 history?"
        
        import time
        start = time.time()
        
        # Must await the route_query
        result = await router.route_query(query)
        
        end = time.time()
        print(f"\n⏱️ Total Time: {end - start:.2f}s")
        print(f"✅ Routed to: {result['routing'].get('datasource', 'UNKNOWN')}")
        print(f"📚 Sources: {len(result.get('sources', []))}")

    asyncio.run(main())