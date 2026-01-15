"""
Enhanced Intelligent Router with Tavily Integration
====================================================
Routes queries to Tavily for best quality AI-powered search results

Uses new fallback chain with Tavily as PRIMARY:
Tavily → Wikipedia → ArXiv → Google API → Bing API → Mock
"""

from typing import Dict, Any, List, Optional
from query_classifier import QueryClassifier
from web_search_tavily import WebSearchEnhanced, HybridSearchEnhanced
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#IntelligentSourceRouter
class IntelligentSourceRouter:
    """
    Intelligent Source Router with Tavily Integration
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_web_search: bool = True,
        web_max_results: int = 5,
        fetch_full_content: bool = True
    ):
        """
        Initialize the enhanced router with Tavily
        
        Args:
            api_key: OpenAI API key
            enable_web_search: Whether to enable web search
            web_max_results: Maximum web search results
            fetch_full_content: Whether to fetch full page content
        """
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
        
        # Initialize enhanced web search with Tavily
        if enable_web_search:
            try:
                self.web_search = WebSearchEnhanced(max_results=web_max_results)
                self.hybrid_search = HybridSearchEnhanced(self.web_search)
                logger.info("✅ Enhanced Web Search (Tavily) and Hybrid Search initialized")
            except Exception as e:
                logger.warning(f"⚠️ Web search initialization failed: {e}")
                self.web_search = None
                self.hybrid_search = None
        else:
            self.web_search = None
            self.hybrid_search = None
        
        self.routing_history = []
    
    def route_query(
        self,
        query: str,
        local_retriever=None,
        has_uploaded_docs: bool = False
    ) -> Dict[str, Any]:
        """
        Route a query with Tavily as primary backend
        
        Returns dict with routing decision and results
        """
        logger.info(f"🔄 Routing query: {query[:50]}...")
        
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
            # Step 1: Classify query
            if self.query_classifier is None:
                logger.warning("⚠️ Query classifier not available, using default routing")
                datasource = self._fallback_datasource_selection(has_uploaded_docs, query)
                result['routing']['datasource'] = datasource
                result['routing']['reasoning'] = "Default routing (classifier unavailable)"
                result['routing']['confidence'] = 0.5
            else:
                classification = self.query_classifier.classify_query(
                    query,
                    has_uploaded_docs
                )
                result['routing'] = classification
            
            # Step 2: Execute retrieval based on classification
            datasource = result['routing']['datasource']
            logger.info(f"📍 Selected datasource: {datasource}")
            
            retrieval_result = self._retrieve_hybrid(query, local_retriever)
            result.update(retrieval_result)

            # Execute appropriate retrieval method
            if datasource == 'local_rag':
                retrieval_result = self._retrieve_local(query, local_retriever)
                result.update(retrieval_result)
            elif datasource == 'web_search':
                retrieval_result = self._retrieve_web(query)
                result.update(retrieval_result)
            elif datasource == 'hybrid':
                retrieval_result = self._retrieve_hybrid(query, local_retriever)
                result.update(retrieval_result)
            else:
                logger.warning(f"Unknown datasource: {datasource}, defaulting to web_search")
                retrieval_result = self._retrieve_web(query)
                result.update(retrieval_result)

            # Step 3: Save routing history
            self._save_routing_history(result)
            
            logger.info(f"✅ Query routed successfully to {datasource}")
            
        except Exception as e:
            logger.error(f"❌ Routing error: {e}", exc_info=True)
            result['error'] = str(e)
            result['context'] = f"Error during routing: {str(e)}"
        
        return result
    
    # 在 IntelligentSourceRouter 类中添加以下方法

    def _rerank_documents(self, query: str, documents: List, top_k: int = 4) -> List:
        """
        使用重排序模型对检索到的文档进行重新排序
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            top_k: 返回前K个最相关的文档
        
        Returns:
            重排序后的文档列表
        """
        if len(documents) <= 1:
            return documents
        
        try:
            # 方法1: 使用交叉编码器（推荐）
            return self._rerank_with_cross_encoder(query, documents, top_k)
            
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed, using BM25 fallback: {e}")
            # 方法2: 使用BM25作为备选
            return self._rerank_with_bm25(query, documents, top_k)

    def _rerank_with_cross_encoder(self, query: str, documents: List, top_k: int) -> List:
        """使用交叉编码器进行重排序"""
        try:
            from sentence_transformers import CrossEncoder
            
            # 初始化交叉编码器模型
            if not hasattr(self, 'cross_encoder'):
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # 准备输入对 (query, document)
            pairs = [(query, doc.page_content) for doc in documents]
            
            # 计算相关性分数
            scores = self.cross_encoder.predict(pairs)
            
            # 根据分数排序文档
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # 返回前top_k个文档
            reranked_docs = [doc for score, doc in scored_docs[:top_k]]
            
            logger.info(f"🔀 Cross-encoder reranking completed: {len(reranked_docs)} docs selected")
            return reranked_docs
            
        except ImportError:
            logger.error("sentence-transformers not installed, please install: pip install sentence-transformers")
            return documents[:top_k]

    def _rerank_with_bm25(self, query: str, documents: List, top_k: int) -> List:
        """使用BM25算法进行重排序（备选方案）"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba  # 中文分词，英文可以使用 nltk
            
            # 分词处理
            tokenized_docs = []
            for doc in documents:
                # 英文分词（如果是中文，使用 jieba.cut）
                tokens = doc.page_content.lower().split()  
                tokenized_docs.append(tokens)
            
            # 初始化BM25
            bm25 = BM25Okapi(tokenized_docs)
            
            # 查询分词
            query_tokens = query.lower().split()
            
            # 计算BM25分数
            doc_scores = bm25.get_scores(query_tokens)
            
            # 根据分数排序
            scored_docs = list(zip(doc_scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # 返回前top_k个文档
            reranked_docs = [doc for score, doc in scored_docs[:top_k]]
            
            logger.info(f"🔀 BM25 reranking completed: {len(reranked_docs)} docs selected")
            return reranked_docs
            
        except ImportError:
            logger.error("rank_bm25 not installed, using original order")
            return documents[:top_k]

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
    
    def _detect_multi_part_query(self, query: str) -> List[str]:
        """
        Detect if query has multiple parts that might need separate searches
        
        Examples:
        - "Who won X and what is Y?" → ["Who won X", "what is Y"]
        - "What is X and Y?" → ["What is X and Y"] (single search)
        
        Returns:
            List of query parts, or empty list if single-part query
        """
        query_lower = query.lower()
        
        # Common multi-part connectors
        multi_part_indicators = [
            ' and what is',
            ' and what are',
            ' and what was',
            ' and who is',
            ' and who are',
            ' and where is',
            ' and when is',
            ' and how is',
            ' and what is the',
            ' and what are the',
            ', and what',
            ', and who',
            ', and where',
            ', and when',
            ', and how'
        ]
        
        parts = []
        for indicator in multi_part_indicators:
            if indicator in query_lower:
                # Split on the indicator
                split_parts = query.split(indicator, 1)
                if len(split_parts) == 2:
                    parts = [
                        split_parts[0].strip(),
                        (indicator + split_parts[1]).strip()
                    ]
                    break
        
        # If no multi-part detected, return empty list (single query)
        return parts if parts else []
    
    def _retrieve_web(self, query: str) -> Dict[str, Any]:
        """
        Retrieve from web search (now with Tavily as primary)
        
        Fallback chain: Tavily → Wikipedia → ArXiv → Google → Bing → Mock
        """
        logger.info("🌐 Performing enhanced web search (Tavily primary)...")
        
        if self.web_search is None:
            logger.warning("⚠️ Web search not enabled")
            return {
                'context': 'Web search is not enabled.',
                'sources': [],
                'retrieval_type': 'web_search',
                'error': 'Web search not enabled'
            }
        
        try:
            logger.info(f"🔍 Searching for: {query}")
            
            # Check if query mentions specific information that needs targeted search
            # (e.g., "Douban score", "IMDB rating", etc.)
            targeted_searches = self._identify_targeted_searches(query)
            
            # Check if query asks for historical/multi-year data
            historical_searches = self._identify_historical_searches(query)
            
            # Use enhanced search with Tavily as primary
            web_results = self.web_search.search_and_extract(
                query,
                fetch_content=self.fetch_full_content
            )
            
            # Perform historical searches for multi-year queries
            if historical_searches:
                logger.info(f"🔍 Performing historical searches: {historical_searches[:2]}")
                for search_term in historical_searches[:2]:  # Limit to 2 additional searches
                    additional_results = self.web_search.search_and_extract(
                        search_term,
                        fetch_content=self.fetch_full_content
                    )
                    # Merge results, avoiding duplicates
                    existing_urls = {r.get('url', '') for r in web_results}
                    for result in additional_results:
                        if result.get('url', '') not in existing_urls:
                            web_results.append(result)
                            if len(web_results) >= 10:  # Limit total results
                                break
                    if len(web_results) >= 10:
                        break
            
            # Perform targeted searches for specific information requests
            if targeted_searches:
                logger.info(f"🔍 Performing targeted searches: {targeted_searches}")
                for search_term in targeted_searches:
                    additional_results = self.web_search.search_and_extract(
                        search_term,
                        fetch_content=self.fetch_full_content
                    )
                    # Merge results, avoiding duplicates
                    existing_urls = {r.get('url', '') for r in web_results}
                    for result in additional_results:
                        if result.get('url', '') not in existing_urls:
                            web_results.append(result)
                            if len(web_results) >= 10:  # Limit total results
                                break
                    if len(web_results) >= 10:
                        break
            
            # Check if query has multiple parts (e.g., "who won X and what is Y")
            # If so, we might need multiple searches
            query_parts = self._detect_multi_part_query(query)
            
            # If query has multiple parts and we got limited results, try additional searches
            if query_parts and len(query_parts) > 1 and len(web_results) < 3:
                logger.info(f"🔍 Multi-part query detected, performing additional searches for missing parts")
                for part in query_parts[1:]:  # Skip first part (already searched)
                    if part and len(part.strip()) > 5:  # Only search meaningful parts
                        additional_results = self.web_search.search_and_extract(
                            part,
                            fetch_content=self.fetch_full_content
                        )
                        # Merge results, avoiding duplicates
                        existing_urls = {r.get('url', '') for r in web_results}
                        for result in additional_results:
                            if result.get('url', '') not in existing_urls:
                                web_results.append(result)
                                if len(web_results) >= 10:  # Limit total results
                                    break
                        if len(web_results) >= 10:
                            break
            
            if not web_results:
                logger.info("ℹ️ No web search results found")
                return {
                    'context': 'No web search results found for your query.',
                    'sources': [],
                    'retrieval_type': 'web_search',
                    'num_results': 0,
                    'raw_search_results': []
                }
            
            # Build context with results
            context = f"=== 🌐 WEB SEARCH RESULTS (Tavily Enhanced, As of {datetime.now().strftime('%B %d, %Y')}) ===\n\n"
            context += f"Search Query: {query}\n"
            context += "=" * 70 + "\n\n"
            
            sources = []
            
            for i, result in enumerate(web_results, 1):
                title = result.get('title', 'Unknown')
                url = result.get('url', '')
                snippet = result.get('snippet', '')
                full_content = result.get('full_content', snippet)
                source = result.get('source', 'Web Search')
                result_type = result.get('type', 'web')
                is_openai = result.get('is_openai_result', False)
                is_direct_answer = result.get('type') == 'direct_answer' or (not url and is_openai)
                
                content_to_use = full_content if full_content else snippet
                
                # Highlight Direct Answers and OpenAI results prominently
                if is_direct_answer:
                    context += f"⭐ [DIRECT ANSWER - HIGH PRIORITY] ⭐\n"
                elif is_openai:
                    context += f"🔵 [OPENAI WEB SEARCH RESULT - HIGH PRIORITY] 🔵\n"
                else:
                    context += f"[Result {i}] ({source} - {result_type})\n"
                
                context += f"Title: {title}\n"
                
                if url:
                    context += f"URL: {url}\n"
                
                context += f"Content:\n{content_to_use}\n"
                context += "-" * 70 + "\n\n"
                
                sources.append({
                    'type': 'web',
                    'title': title,
                    'url': url,
                    'snippet': snippet,
                    'full_content': content_to_use,
                    'source': source,
                    'result_type': result_type
                })
            
            logger.info(f"✅ Retrieved {len(web_results)} web search results")
            
            return {
                'context': context,
                'sources': sources,
                'retrieval_type': 'web_search',
                'num_results': len(web_results),
                'raw_search_results': web_results
            }
            
        except Exception as e:
            logger.error(f"❌ Error performing web search: {e}", exc_info=True)
            return {
                'context': f'Error performing web search: {str(e)}',
                'sources': [],
                'retrieval_type': 'web_search',
                'error': str(e),
                'raw_search_results': []
            }
    
    def _retrieve_local(
        self,
        query: str,
        local_retriever
    ) -> Dict[str, Any]:
        """Retrieve from local RAG system"""
        logger.info("📄 Retrieving from local documents...")
        
        if local_retriever is None:
            logger.warning("⚠️ No local retriever configured")
            return {
                'context': 'No local documents available.',
                'sources': [],
                'retrieval_type': 'local_rag',
                'error': 'No local retriever configured'
            }
        
        try:
            docs = local_retriever.get_relevant_documents(query)
            
            if not docs:
                logger.info("ℹ️ No relevant local documents found")
                return {
                    'context': 'No relevant documents found in uploaded files.',
                    'sources': [],
                    'retrieval_type': 'local_rag',
                    'num_results': 0
                }
            
            # Reranking
            if len(docs) > 1:
                try:
                    docs = self._rerank_documents(query, docs, top_k=4)
                    logger.info(f"✅ Reranked {len(docs)} documents, selected top 4")
                except Exception as e:
                    logger.warning(f"⚠️ Reranking failed, using original order: {e}")
            
            context = "=== 📄 LOCAL DOCUMENT RESULTS ===\n\n"
            sources = []
            
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                chunk = doc.metadata.get('chunk', 'N/A')
                
                context += f"[Document {i}] {source}\n"
                
                if page != 'N/A':
                    context += f"Page: {page} | "
                if chunk != 'N/A':
                    context += f"Chunk: {chunk}\n"
                
                context += f"Content: {doc.page_content}\n\n"
                
                sources.append({
                    'type': 'local',
                    'source': source,
                    'page': page,
                    'chunk': chunk,
                    'content': doc.page_content
                })
            
            logger.info(f"✅ Retrieved {len(docs)} local documents")
            
            return {
                'context': context,
                'sources': sources,
                'retrieval_type': 'local_rag',
                'num_results': len(docs),
                'reranked': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error retrieving local documents: {e}", exc_info=True)
            return {
                'context': f'Error retrieving local documents: {str(e)}',
                'sources': [],
                'retrieval_type': 'local_rag',
                'error': str(e)
            }
    
    def _retrieve_hybrid(
        self,
        query: str,
        local_retriever
    ) -> Dict[str, Any]:
        """Retrieve from both local and web sources (with Tavily)"""
        logger.info("🔄 Performing hybrid retrieval (local + web with Tavily)...")
        
        try:
            # Get local results
            local_results = []
            if local_retriever:
                try:
                    local_docs = local_retriever.get_relevant_documents(query)
                    # Reranking
                    if len(local_docs) > 1:
                        try:
                            local_docs = self._rerank_documents(query, local_docs, top_k=4)
                            logger.info(f"✅ Reranked {len(local_docs)} documents, selected top 4")
                        except Exception as e:
                            logger.warning(f"⚠️ Reranking failed, using original order: {e}")

                    local_results = [
                        {
                            'source': doc.metadata.get('source', 'Unknown'),
                            'page': doc.metadata.get('page', 'N/A'),
                            'chunk': doc.metadata.get('chunk', 'N/A'),
                            'content': doc.page_content
                        }
                        for doc in local_docs
                    ]
                    logger.info(f"✅ Retrieved {len(local_results)} local results")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to retrieve local results: {e}")
            
            # Get web results (with Tavily primary)
            web_results = []
            if self.web_search:
                try:
                    web_results = self.web_search.search_and_extract(
                        query,
                        fetch_content=self.fetch_full_content
                    )
                    logger.info(f"✅ Retrieved {len(web_results)} web results")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to retrieve web results: {e}")
            
            # Format hybrid context
            hybrid_context = f"=== 🔄 HYBRID RETRIEVAL RESULTS (As of {datetime.now().strftime('%B %d, %Y')}) ===\n\n"
            
            # Local section
            hybrid_context += "📄 LOCAL DOCUMENTS:\n"
            hybrid_context += "-" * 70 + "\n"
            
            if local_results:
                for i, result in enumerate(local_results, 1):
                    hybrid_context += f"[Local {i}] {result['source']}\n"
                    if result['page'] != 'N/A':
                        hybrid_context += f"Page: {result['page']} | "
                    if result['chunk'] != 'N/A':
                        hybrid_context += f"Chunk: {result['chunk']}\n"
                    hybrid_context += f"Content: {result['content']}\n\n"
            else:
                hybrid_context += "No local documents found.\n\n"
            
            # Web section
            hybrid_context += "\n🌐 WEB SEARCH RESULTS (Tavily Enhanced):\n"
            hybrid_context += "-" * 70 + "\n"
            
            if web_results:
                for i, result in enumerate(web_results, 1):
                    title = result.get('title', 'Unknown')
                    url = result.get('url', '')
                    content = result.get('full_content', result.get('snippet', ''))
                    source = result.get('source', 'Web Search')
                    
                    hybrid_context += f"[Web {i}] {title} ({source})\n"
                    if url:
                        hybrid_context += f"URL: {url}\n"
                    hybrid_context += f"Content: {content}\n\n"
            else:
                hybrid_context += "No web results found.\n\n"
            
            # Combine sources
            all_sources = []
            for result in local_results:
                all_sources.append({
                    'type': 'local',
                    'source': result['source'],
                    'page': result['page'],
                    'chunk': result['chunk'],
                    'content': result['content']
                })
            
            for result in web_results:
                all_sources.append({
                    'type': 'web',
                    'title': result.get('title', 'Unknown'),
                    'url': result.get('url', ''),
                    'content': result.get('full_content', result.get('snippet', '')),
                    'source': result.get('source', 'Web Search')
                })
            
            logger.info(f"✅ Hybrid retrieval: {len(local_results)} local + {len(web_results)} web")
            
            return {
                'context': hybrid_context,
                'sources': all_sources,
                'retrieval_type': 'hybrid',
                'num_local_results': len(local_results),
                'num_web_results': len(web_results),
                'total_results': len(local_results) + len(web_results),
                'raw_search_results': web_results
            }
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid retrieval: {e}", exc_info=True)
            return {
                'context': f'Error in hybrid retrieval: {str(e)}',
                'sources': [],
                'retrieval_type': 'hybrid',
                'error': str(e),
                'raw_search_results': []
            }
    
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


# ==================== TEST ====================

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED INTELLIGENT ROUTER - TAVILY TEST")
    print("=" * 80)
    
    try:
        router = IntelligentSourceRouterTavily(
            enable_web_search=True,
            web_max_results=3,
            fetch_full_content=True
        )
        
        test_queries = [
            "What are the latest AI developments in 2025?",
            "Explain machine learning to beginners",
            "Current Python programming trends"
        ]
        
        print("\n🔄 Testing enhanced router with Tavily...\n")
        
        for query in test_queries:
            print(f"{'='*80}")
            print(f"Query: {query}")
            print('='*80)
            
            result = router.route_query(query, local_retriever=None, has_uploaded_docs=False)
            
            print(f"✅ Routed to: {result['routing']['datasource'].upper()}")
            print(f"📝 Reasoning: {result['routing']['reasoning']}")
            print(f"📊 Confidence: {result['routing']['confidence']:.2%}")
            print(f"📚 Sources found: {len(result.get('sources', []))}")
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                context_preview = result.get('context', '')[:300]
                print(f"\n📄 Context Preview:\n{context_preview}...")
        
        print(f"\n{'='*80}")
        print("ROUTING STATISTICS")
        print('='*80)
        
        stats = router.get_routing_stats()
        print(f"Total queries: {stats['total_queries']}")
        print(f"By source: {stats['by_source']}")
        print(f"Average confidence: {stats['avg_confidence']:.2%}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        
        print(f"\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()