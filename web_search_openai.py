"""
OpenAI Web Search Integration

Integrates OpenAI's web search tool from the Responses API to handle
real-time queries like bus schedules, stock prices, weather, etc.
"""

from typing import List, Dict, Any, Optional
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIWebSearchIntegration:
    """
    OpenAI Web Search integration using the Responses API
    
    Supports three modes:
    1. Non-reasoning web search (fast, quick lookups)
    2. Agentic search with reasoning (thorough, for complex queries)
    3. Deep research (very thorough, for extended investigations)
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", reasoning_level: str = "low"):
        """
        Initialize OpenAI Web Search
        
        Args:
            api_key: OpenAI API key (from environment if not provided)
            model: Model to use (gpt-4o, gpt-5, o3-deep-research, etc.)
            reasoning_level: Reasoning effort ("low", "medium", "high")
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
            self.model = model
            self.reasoning_level = reasoning_level
            
            if self.client.api_key:
                logger.info(f"✅ OpenAI Web Search initialized (model: {model}, reasoning: {reasoning_level})")
            else:
                logger.warning("⚠️ OpenAI API key not found")
        except ImportError:
            logger.error("❌ OpenAI package not installed. Install with: pip install openai>=1.10.0")
            self.client = None
        except Exception as e:
            logger.error(f"❌ Error initializing OpenAI client: {e}")
            self.client = None
    
    def search(
        self, 
        query: str, 
        max_results: int = 5,
        reasoning_level: Optional[str] = None,
        user_location: Optional[Dict[str, str]] = None,
        domain_filters: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform web search using OpenAI's web search tool
        
        Args:
            query: Search query
            max_results: Maximum results to return (not directly controllable, but used for formatting)
            reasoning_level: Override default reasoning level ("low", "medium", "high")
            user_location: Optional location dict with country, city, region, timezone
            domain_filters: Optional list of allowed domains (max 20)
            
        Returns:
            List of search results with citations
        """
        if not self.client:
            logger.warning("⚠️ OpenAI client not initialized, skipping web search")
            return []
        
        try:
            # Determine reasoning configuration
            reasoning = reasoning_level or self.reasoning_level
            
            # Build tools configuration
            web_search_tool = {
                "type": "web_search"
            }
            
            # Add domain filters if provided
            if domain_filters:
                web_search_tool["filters"] = {
                    "allowed_domains": domain_filters[:20]  # Max 20 domains
                }
            
            # Add user location if provided
            if user_location:
                web_search_tool["user_location"] = {
                    "type": "approximate",
                    **user_location
                }
            
            # Configure reasoning based on level
            reasoning_config = None
            if reasoning in ["low", "medium", "high"]:
                reasoning_config = {"effort": reasoning}
            
            logger.info(f"🔍 [OpenAI Web Search] Query: {query} (reasoning: {reasoning})")
            
            # Try Responses API first (newer API)
            try:
                if hasattr(self.client, 'responses'):
                    response = self.client.responses.create(
                        model=self.model,
                        tools=[web_search_tool],
                        reasoning=reasoning_config,
                        input=query
                    )
                    results = self._extract_results_from_response(response, query)
                    logger.info(f"✅ [OpenAI Web Search] Retrieved {len(results)} results")
                    return results
            except (AttributeError, Exception) as e:
                logger.debug(f"Responses API not available or failed: {e}")
            
            # Fallback: Use Chat Completions API with web search models
            # Models: gpt-5-search-api, gpt-4o-search-preview, gpt-4o-mini-search-preview
            try:
                logger.info("⚠️ Using Chat Completions API with search model as fallback")
                if "gpt-5" in self.model.lower():
                    search_model = "gpt-5-search-api"
                elif "gpt-4" in self.model.lower():
                    search_model = "gpt-4o-search-preview"
                else:
                    search_model = "gpt-4o-mini-search-preview"
                
                response = self.client.chat.completions.create(
                    model=search_model,
                    messages=[
                        {"role": "user", "content": query}
                    ]
                )
                results = self._extract_from_chat_completion(response, query)
                logger.info(f"✅ [OpenAI Web Search] Retrieved {len(results)} results (via Chat Completions)")
                return results
            except Exception as e:
                logger.error(f"❌ Chat Completions fallback also failed: {e}")
                return []
            
        except Exception as e:
            logger.error(f"❌ Error in OpenAI web search: {e}", exc_info=True)
            return []
    
    def _extract_from_chat_completion(self, response, query: str) -> List[Dict[str, Any]]:
        """
        Extract results from Chat Completions API response (fallback)
        
        Args:
            response: Chat Completions API response
            query: Original query
            
        Returns:
            List of formatted search results
        """
        results = []
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                if content:
                    results.append({
                        'title': 'OpenAI Web Search Result',
                        'url': '',
                        'snippet': content,
                        'full_content': content,
                        'source': 'OpenAI Web Search',
                        'type': 'answer',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Try to extract citations if available
                    if hasattr(response.choices[0].message, 'citations'):
                        citations = response.choices[0].message.citations
                        if citations:
                            for citation in citations:
                                if isinstance(citation, dict):
                                    results.append({
                                        'title': citation.get('title', 'Citation'),
                                        'url': citation.get('url', ''),
                                        'snippet': citation.get('snippet', ''),
                                        'source': 'OpenAI Web Search',
                                        'type': 'web',
                                        'timestamp': datetime.now().isoformat()
                                    })
        except Exception as e:
            logger.error(f"Error extracting from chat completion: {e}")
        
        return results
    
    def _extract_results_from_response(self, response, query: str) -> List[Dict[str, Any]]:
        """
        Extract search results from OpenAI Responses API response
        
        Args:
            response: OpenAI Responses API response object
            query: Original query
            
        Returns:
            List of formatted search results
        """
        results = []
        
        try:
            # Get the final message content
            message_content = None
            sources = []
            citations = []
            
            # Parse response items
            if hasattr(response, 'items'):
                for item in response.items:
                    # Extract web search call information
                    if hasattr(item, 'type') and item.type == 'web_search_call':
                        if hasattr(item, 'action') and hasattr(item.action, 'sources'):
                            sources = item.action.sources or []
                    
                    # Extract message content
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content'):
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    message_content = content_item.text
                                
                                # Extract citations
                                if hasattr(content_item, 'annotations'):
                                    for annotation in content_item.annotations:
                                        if hasattr(annotation, 'type') and annotation.type == 'url_citation':
                                            citations.append({
                                                'url': getattr(annotation, 'url', ''),
                                                'title': getattr(annotation, 'title', ''),
                                                'start_index': getattr(annotation, 'start_index', 0),
                                                'end_index': getattr(annotation, 'end_index', 0)
                                            })
            
            # If we have message content, create a result from it
            if message_content:
                # Create main result from the response text
                results.append({
                    'title': 'OpenAI Web Search Result',
                    'url': '',
                    'snippet': message_content,
                    'full_content': message_content,
                    'source': 'OpenAI Web Search',
                    'type': 'answer',
                    'citations': citations,
                    'sources': sources,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Add individual sources as separate results
            for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources
                if isinstance(source, str):
                    results.append({
                        'title': f'Source {i}',
                        'url': source,
                        'snippet': f'Source from OpenAI web search: {source}',
                        'source': 'OpenAI Web Search',
                        'type': 'web',
                        'timestamp': datetime.now().isoformat()
                    })
                elif isinstance(source, dict):
                    results.append({
                        'title': source.get('title', f'Source {i}'),
                        'url': source.get('url', ''),
                        'snippet': source.get('snippet', ''),
                        'source': 'OpenAI Web Search',
                        'type': 'web',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # If no results but we have citations, create results from citations
            if not results and citations:
                for i, citation in enumerate(citations[:10], 1):
                    results.append({
                        'title': citation.get('title', f'Citation {i}'),
                        'url': citation.get('url', ''),
                        'snippet': f'Cited source: {citation.get("title", "")}',
                        'source': 'OpenAI Web Search',
                        'type': 'web',
                        'timestamp': datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"❌ Error extracting results from OpenAI response: {e}", exc_info=True)
        
        return results if results else []
    
    def search_fast(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Fast web search using non-reasoning model (quick lookups)
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of search results
        """
        return self.search(query, max_results, reasoning_level="low")
    
    def search_thorough(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Thorough web search using reasoning model (complex queries)
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of search results
        """
        return self.search(query, max_results, reasoning_level="medium")
    
    def search_deep(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Deep research using high reasoning (extended investigations)
        
        Note: This can take several minutes
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of search results
        """
        return self.search(query, max_results, reasoning_level="high")


# Convenience function for quick integration
def create_openai_web_search(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    reasoning_level: str = "low"
) -> Optional[OpenAIWebSearchIntegration]:
    """
    Create OpenAI Web Search integration instance
    
    Args:
        api_key: OpenAI API key
        model: Model to use
        reasoning_level: Reasoning effort level
        
    Returns:
        OpenAIWebSearchIntegration instance or None if initialization fails
    """
    try:
        return OpenAIWebSearchIntegration(api_key, model, reasoning_level)
    except Exception as e:
        logger.error(f"Failed to create OpenAI web search: {e}")
        return None

