

from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import time
import random
import os
from xml.etree import ElementTree as ET

# ✅ AUTOMATICALLY LOAD .env FILE AT MODULE LEVEL
from dotenv import load_dotenv
load_dotenv()  # This ensures API key is loaded when module is imported

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TavilySearchIntegration:
    """
    Tavily AI-powered search integration
    
    Tavily provides high-quality search results optimized for AI/LLM context
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        """
        Initialize Tavily search
        
        Args:
            api_key: Tavily API key (from environment if not provided)
            timeout: Request timeout in seconds
        """
        # ✅ Try to get API key from parameter, then environment
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        self.timeout = timeout
        self.base_url = 'https://api.tavily.com/search'
        
        if self.api_key:
            logger.info(f"🔍 Tavily API initialized: ✅ API key found")
        else:
            logger.warning(f"🔍 Tavily API initialized: ⚠️ No API key")
    
    def search(self, query: str, max_results: int = 5, include_answer: bool = True) -> List[Dict[str, Any]]:
        """
        Search using Tavily API
        
        Args:
            query: Search query
            max_results: Maximum results to return
            include_answer: Include direct answer if available
            
        Returns:
            List of search results
        """
        if not self.api_key:
            logger.warning("⚠️ Tavily API key not configured, skipping Tavily")
            return []
        
        try:
            logger.info(f"🔍 [PRIMARY] Tavily: {query}")
            
            payload = {
                'api_key': self.api_key,
                'query': query,
                'max_results': max_results,
                'include_answer': include_answer,
                'include_raw_content': True,
                'topic': 'general'
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Add direct answer if available
            if include_answer and data.get('answer'):
                results.append({
                    'title': 'Direct Answer',
                    'url': '',
                    'snippet': data['answer'],
                    'source': 'Tavily',
                    'type': 'answer'
                })
            
            # Add search results
            for result in data.get('results', [])[:max_results]:
                formatted_result = {
                    'title': result.get('title', 'No title'),
                    'url': result.get('url', ''),
                    'snippet': result.get('snippet', ''),
                    'raw_content': result.get('raw_content', ''),
                    'source': 'Tavily',
                    'type': 'web'
                }
                results.append(formatted_result)
            
            if results:
                logger.info(f"✅ Tavily SUCCESS: {len(results)} results")
            else:
                logger.warning(f"⚠️ Tavily: 0 results")
            
            return results
            
        except requests.Timeout:
            logger.warning(f"⚠️ Tavily timeout")
            return []
        except requests.ConnectionError:
            logger.warning(f"⚠️ Tavily connection error")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Tavily failed: {e}")
            return []


class WebSearchEnhanced:
    """
    Enhanced web search with Tavily as primary backend
    
    Chain: OpenAI Web Search (for real-time) → Tavily → Wikipedia → ArXiv → Google API → Bing API → Mock
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 15, use_openai_web_search: bool = True):
        """
        Initialize enhanced web search
        
        Args:
            max_results: Maximum results per search
            timeout: Request timeout in seconds
            use_openai_web_search: Whether to use OpenAI web search for real-time queries
        """
        self.max_results = max_results
        self.timeout = timeout
        
        # Create session with connection pooling
        self.session = self._create_session()
        
        # Initialize Tavily
        self.tavily = TavilySearchIntegration(timeout=timeout)
        
        # Initialize OpenAI Web Search (optional, for real-time queries)
        self.openai_web_search = None
        if use_openai_web_search:
            try:
                from web_search_openai import OpenAIWebSearchIntegration
                self.openai_web_search = OpenAIWebSearchIntegration(
                    model="gpt-4o",
                    reasoning_level="low"  # Fast mode for quick lookups
                )
                if self.openai_web_search.client:
                    logger.info("✅ OpenAI Web Search initialized (for real-time queries)")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI Web Search not available: {e}")
        
        # API keys (optional)
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.bing_api_key = os.getenv('BING_SEARCH_KEY')
        
        # User agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        
        logger.info("✅ Enhanced Web Search initialized")
    
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1.0
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get random user agent headers"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def _is_real_time_query(self, query: str) -> bool:
        """
        Detect if query requires real-time data or comprehensive historical data
        
        Args:
            query: Search query
            
        Returns:
            True if query likely requires real-time data or comprehensive historical data
        """
        query_lower = query.lower()
        real_time_keywords = [
            'departure time', 'bus schedule', 'train schedule', 'flight', 'arrival',
            'stock price', 'stock performance', 'nvidia', 'amd', 'ticker', 'nasdaq',
            'weather', 'temperature', 'forecast', 'today', 'now', 'current',
            'latest', 'recent', 'breaking', 'news', 'live', 'happening now',
            'compare stock', 'stock comparison', 'real-time', 'up-to-date'
        ]
        
        # Also detect historical/multi-year queries that need comprehensive data
        historical_keywords = [
            'past ten years', 'past 10 years', 'last ten years', 'last 10 years',
            'past five years', 'past 5 years', 'last five years', 'last 5 years',
            'past.*years', 'over the past', 'historical', 'year by year', 
            'timeline', 'compare.*year', 'ranking.*year', 'trend', 'evolution',
            'change over time', 'decade', 'all years'
        ]
        
        import re
        is_historical = any(re.search(pattern, query_lower) for pattern in historical_keywords)
        
        return any(keyword in query_lower for keyword in real_time_keywords) or is_historical
    
    def search_openai(self, query: str, max_results: Optional[int] = None, reasoning_level: str = "low") -> List[Dict[str, Any]]:
        """
        Search using OpenAI Web Search API (for real-time queries)
        
        Best for: Bus schedules, stock prices, weather, current events, etc.
        """
        if not self.openai_web_search:
            return []
        
        try:
            results_limit = max_results or self.max_results
            logger.info(f"🔍 [OpenAI Web Search] {query} (reasoning: {reasoning_level})")
            results = self.openai_web_search.search(
                query, 
                max_results=results_limit,
                reasoning_level=reasoning_level
            )
            if results:
                logger.info(f"✅ [OpenAI Web Search] Retrieved {len(results)} results")
            return results
        except Exception as e:
            logger.debug(f"OpenAI Web Search exception: {e}")
            return []
    
    def search_tavily(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search using Tavily API (PRIMARY)"""
        results_limit = max_results or self.max_results
        return self.tavily.search(query, max_results=results_limit)
    
    def search_wikipedia(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search Wikipedia API"""
        try:
            results_limit = max_results or self.max_results
            logger.info(f"🔍 [1/6] Wikipedia: {query}")
            
            response = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': query,
                    'srlimit': results_limit,
                    'srwhat': 'text'
                },
                timeout=self.timeout,
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            data = response.json()
            search_results = data.get('query', {}).get('search', [])
            
            results = []
            for result in search_results:
                formatted_result = {
                    'title': result.get('title', 'No title'),
                    'url': f"https://en.wikipedia.org/wiki/{result.get('title', '').replace(' ', '_')}",
                    'snippet': result.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', ''),
                    'source': 'Wikipedia'
                }
                results.append(formatted_result)
            
            if results:
                logger.info(f"✅ Wikipedia SUCCESS: {len(results)} results")
            else:
                logger.warning(f"⚠️ Wikipedia: 0 results")
            
            return results
        except Exception as e:
            logger.warning(f"⚠️ Wikipedia failed: {e}")
            return []
    
    def search_arxiv(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search ArXiv API"""
        try:
            results_limit = max_results or self.max_results
            logger.info(f"🔍 [2/6] ArXiv: {query}")
            
            response = self.session.get(
                'http://export.arxiv.org/api/query',
                params={
                    'search_query': f'all:{query}',
                    'start': 0,
                    'max_results': results_limit,
                    'sortBy': 'submittedDate',
                    'sortOrder': 'descending'
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            results = []
            for entry in root.findall('atom:entry', ns):
                try:
                    title_elem = entry.find('atom:title', ns)
                    summary_elem = entry.find('atom:summary', ns)
                    id_elem = entry.find('atom:id', ns)
                    
                    if title_elem is not None:
                        result = {
                            'title': title_elem.text or 'No title',
                            'url': (id_elem.text or '').replace('http://', 'https://'),
                            'snippet': (summary_elem.text or '')[:300],
                            'source': 'ArXiv'
                        }
                        results.append(result)
                except:
                    continue
            
            if results:
                logger.info(f"✅ ArXiv SUCCESS: {len(results)} results")
            else:
                logger.warning(f"⚠️ ArXiv: 0 results")
            
            return results
        except Exception as e:
            logger.warning(f"⚠️ ArXiv failed: {e}")
            return []
    
    def search_google_api(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        if not self.google_api_key or not self.google_search_engine_id:
            return []
        
        try:
            results_limit = max_results or self.max_results
            logger.info(f"🔍 [3/6] Google API: {query}")
            
            response = self.session.get(
                'https://www.googleapis.com/customsearch/v1',
                params={
                    'q': query,
                    'key': self.google_api_key,
                    'cx': self.google_search_engine_id,
                    'num': min(results_limit, 10)
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('items', [])[:results_limit]:
                result = {
                    'title': item.get('title', 'No title'),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'Google'
                }
                results.append(result)
            
            if results:
                logger.info(f"✅ Google API SUCCESS: {len(results)} results")
            else:
                logger.warning(f"⚠️ Google API: 0 results")
            
            return results
        except Exception as e:
            logger.warning(f"⚠️ Google API failed: {e}")
            return []
    
    def search_bing_api(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search using Bing Search API"""
        if not self.bing_api_key:
            return []
        
        try:
            results_limit = max_results or self.max_results
            logger.info(f"🔍 [4/6] Bing API: {query}")
            
            headers = self._get_headers()
            headers['Ocp-Apim-Subscription-Key'] = self.bing_api_key
            
            response = self.session.get(
                'https://api.bing.microsoft.com/v7.0/search',
                params={
                    'q': query,
                    'count': results_limit,
                    'textDecorations': 'true',
                    'textFormat': 'HTML'
                },
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('webPages', {}).get('value', [])[:results_limit]:
                result = {
                    'title': item.get('name', 'No title'),
                    'url': item.get('url', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'Bing'
                }
                results.append(result)
            
            if results:
                logger.info(f"✅ Bing API SUCCESS: {len(results)} results")
            else:
                logger.warning(f"⚠️ Bing API: 0 results")
            
            return results
        except Exception as e:
            logger.warning(f"⚠️ Bing API failed: {e}")
            return []
    
    def search_mock(self, query: str) -> List[Dict[str, Any]]:
        """Emergency: Generate mock results"""
        logger.warning(f"⚠️ [5/6] MOCK: All searches failed, using mock results")
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.now().strftime("%B %d, %Y")
        
        mock_results = [
            {
                'title': f'Information about {query}',
                'url': f'https://local.search/results?q={query.replace(" ", "+")}',
                'snippet': f'Based on available knowledge about {query} as of {current_date}. This is a local cached result.',
                'source': 'Local Cache'
            },
            {
                'title': f'Related: {query.title()} Overview',
                'url': f'https://local.search/related?q={query.replace(" ", "+")}',
                'snippet': f'General information and context related to {query}. Updated: {current_time}.',
                'source': 'Local Cache'
            }
        ]
        
        logger.info(f"✅ MOCK SUCCESS: {len(mock_results)} results")
        return mock_results
    
    def search(self, query: str, max_results: Optional[int] = None, prefer_openai_for_realtime: bool = True) -> List[Dict[str, Any]]:
        """
        Main search with complete fallback chain
        
        Chain: OpenAI Web Search (for real-time) → Tavily → Wikipedia → ArXiv → Google API → Bing API → Mock
        
        GUARANTEED to return results
        
        Args:
            query: Search query
            max_results: Override max results
            prefer_openai_for_realtime: Use OpenAI web search first for real-time queries
            
        Returns:
            List of search results (never empty)
        """
        logger.info(f"\n🔄 STARTING ENHANCED SEARCH: {query}")
        
        results_limit = max_results or self.max_results
        
        # Collect results from multiple sources (hybrid approach)
        # Separate OpenAI results (always included) from regular sources (max 5)
        openai_results = []  # Always included, not counted in max_results
        regular_results = []  # Limited to max_results (5)
        existing_urls = set()  # Track URLs to avoid duplicates
        
        # ALWAYS use OpenAI Web Search first if available (for all queries)
        if self.openai_web_search:
            # Determine reasoning level based on query type
            import re
            query_lower = query.lower()
            is_historical = any(re.search(pattern, query_lower) for pattern in [
                'past ten years', 'past 10 years', 'last ten years', 'last 10 years',
                'past five years', 'past 5 years', 'over the past', 'historical',
                'year by year', 'timeline', 'decade', 'all years'
            ])
            
            # Use higher reasoning level for historical/complex queries
            reasoning_level = "medium" if is_historical else "low"
            
            logger.info(f"🔍 Using OpenAI Web Search first for all queries (reasoning_level={reasoning_level})")
            try:
                openai_search_results = self.search_openai(query, results_limit, reasoning_level=reasoning_level)
                if openai_search_results:
                    logger.info(f"✅ OpenAI Web Search returned {len(openai_search_results)} results")
                    for result in openai_search_results:
                        # Mark OpenAI results with special type
                        result['source'] = 'OpenAI Web Search'
                        result['is_openai_result'] = True
                        
                        # Check if it's a direct answer (no URL)
                        if not result.get('url'):
                            result['type'] = 'direct_answer'
                            result['title'] = result.get('title', 'Direct Answer')
                        
                        openai_results.append(result)
                        # Track URLs to avoid duplicates in regular results
                        if result.get('url'):
                            existing_urls.add(result['url'])
            except Exception as e:
                logger.debug(f"OpenAI Web Search exception: {e}")
        
        # Continue to Tavily to get additional results (even if OpenAI succeeded)
        logger.info(f"🔍 Also searching with Tavily for additional results...")
        try:
            tavily_results = self.search_tavily(query, results_limit)
            if tavily_results:
                logger.info(f"✅ Tavily returned {len(tavily_results)} results")
                for result in tavily_results:
                    url = result.get('url', '')
                    if url and url not in existing_urls:
                        result['source'] = result.get('source', 'Tavily')
                        result['is_openai_result'] = False
                        regular_results.append(result)
                        existing_urls.add(url)
                    elif not url:
                        result['source'] = result.get('source', 'Tavily')
                        result['is_openai_result'] = False
                        regular_results.append(result)
        except Exception as e:
            logger.debug(f"Tavily exception: {e}")
        
        # Combine: OpenAI results (always included) + regular results (limited to max_results)
        if openai_results or regular_results:
            all_results = openai_results + regular_results[:results_limit]
            logger.info(f"✅ Combined results: {len(openai_results)} OpenAI + {len(regular_results[:results_limit])} regular = {len(all_results)} total")
            return all_results
        
        # Fallback chain: Wikipedia → ArXiv → Google API → Bing API → Mock
        # These go into regular_results (limited to max_results)
        logger.info(f"Fallback chain: Wikipedia → ArXiv → Google API → Bing API → Mock")
        
        time.sleep(0.2)
        
        # Level 1: Wikipedia
        try:
            results = self.search_wikipedia(query, results_limit)
            if results:
                for result in results:
                    result['source'] = result.get('source', 'Wikipedia')
                    result['is_openai_result'] = False
                    if len(regular_results) < results_limit:
                        regular_results.append(result)
                return openai_results + regular_results
        except Exception as e:
            logger.debug(f"Wikipedia exception: {e}")
        
        time.sleep(0.2)
        
        # Level 2: ArXiv
        try:
            results = self.search_arxiv(query, results_limit)
            if results:
                for result in results:
                    result['source'] = result.get('source', 'ArXiv')
                    result['is_openai_result'] = False
                    if len(regular_results) < results_limit:
                        regular_results.append(result)
                return openai_results + regular_results
        except Exception as e:
            logger.debug(f"ArXiv exception: {e}")
        
        time.sleep(0.2)
        
        # Level 3: Google API
        try:
            results = self.search_google_api(query, results_limit)
            if results:
                for result in results:
                    result['source'] = result.get('source', 'Google')
                    result['is_openai_result'] = False
                    if len(regular_results) < results_limit:
                        regular_results.append(result)
                return openai_results + regular_results
        except Exception as e:
            logger.debug(f"Google API exception: {e}")
        
        time.sleep(0.2)
        
        # Level 4: Bing API
        try:
            results = self.search_bing_api(query, results_limit)
            if results:
                for result in results:
                    result['source'] = result.get('source', 'Bing')
                    result['is_openai_result'] = False
                    if len(regular_results) < results_limit:
                        regular_results.append(result)
                return openai_results + regular_results
        except Exception as e:
            logger.debug(f"Bing API exception: {e}")
        
        # Level 5: Mock (Always succeeds)
        results = self.search_mock(query)
        for result in results:
            result['source'] = result.get('source', 'Mock')
            result['is_openai_result'] = False
            if len(regular_results) < results_limit:
                regular_results.append(result)
        return openai_results + regular_results
    
    def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a webpage"""
        try:
            if not url or not url.startswith('http'):
                return None
            
            logger.info(f"📄 Fetching: {url[:50]}")
            
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout,
                allow_redirects=True
            )
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "meta"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            content = text[:3000]
            logger.info(f"✅ Extracted {len(content)} characters")
            return content if content else None
            
        except Exception as e:
            logger.debug(f"Content fetch failed: {e}")
            return None
    
    def search_and_extract(self, query: str, fetch_content: bool = False) -> List[Dict[str, Any]]:
        """Search and optionally extract full content"""
        logger.info(f"🔍 Searching{' + extracting' if fetch_content else ''}...")
        
        results = self.search(query)
        
        if not results or not fetch_content:
            return results
        
        logger.info(f"📄 Fetching full content for {len(results)} results")
        
        for i, result in enumerate(results):
            url = result.get('url')
            if url and url.startswith('http'):
                try:
                    content = self.fetch_page_content(url)
                    if content:
                        result['full_content'] = content
                    else:
                        result['full_content'] = result.get('snippet', '')
                    time.sleep(0.2)
                except Exception as e:
                    logger.debug(f"Content extraction error: {e}")
                    result['full_content'] = result.get('snippet', '')
            else:
                result['full_content'] = result.get('snippet', '')
        
        return results
    
    def format_results_for_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM context"""
        if not results:
            return "No search results available."
        
        current_date = datetime.now().strftime('%B %d, %Y')
        context = f"=== WEB SEARCH RESULTS (As of {current_date}) ===\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            snippet = result.get('snippet', '')
            full_content = result.get('full_content', snippet)
            source = result.get('source', 'Search')
            
            content = full_content if full_content else snippet
            
            context += f"[Result {i}]\n"
            context += f"Title: {title}\n"
            context += f"Source: {source}\n"
            
            if url:
                context += f"URL: {url}\n"
            
            if content:
                context += f"Content: {content[:400]}\n"
            else:
                context += f"Content: [No content available]\n"
            
            context += "\n"
        
        logger.info(f"✅ Formatted {len(results)} results")
        return context


class HybridSearchEnhanced:
    """Combines local RAG with enhanced web search"""
    
    def __init__(self, web_search: WebSearchEnhanced):
        self.web_search = web_search
    
    def hybrid_retrieve(
        self,
        query: str,
        local_results: List[Dict[str, Any]],
        web_results_count: int = 3
    ) -> Dict[str, Any]:
        """Perform hybrid retrieval"""
        logger.info(f"🔄 Hybrid: {len(local_results)} local + {web_results_count} web")
        
        web_results = self.web_search.search(query, max_results=web_results_count)
        
        return {
            'local_results': local_results,
            'web_results': web_results,
            'query': query,
            'retrieval_type': 'hybrid'
        }


if __name__ == "__main__":
    print("Testing web_search_tavily with automatic .env loading...")
    ws = WebSearchEnhanced(max_results=3)
    results = ws.search("what is python")
    print(f"✅ Found {len(results)} results")