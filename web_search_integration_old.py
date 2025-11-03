"""
Web Search Integration Module
"""

from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import time


class WebSearchIntegration:
    """
    Web search integration using DuckDuckGo for external knowledge retrieval
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        """
        Initialize web search integration
        
        Args:
            max_results: Maximum number of search results to retrieve
            timeout: Request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.ddgs = DDGS()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform web search using DuckDuckGo
        
        Args:
            query: Search query string
            max_results: Override default max results
            
        Returns:
            List of search result dictionaries
        """
        try:
            results_limit = max_results or self.max_results
            
            # Perform search
            results = []
            search_results = self.ddgs.text(
                query,
                max_results=results_limit
            )
            
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'DuckDuckGo'
                })
            
            return results
            
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract text content from a webpage
        
        Args:
            url: URL to fetch
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit to first 5000 characters
            
        except Exception as e:
            print(f"Error fetching page content from {url}: {e}")
            return None
    
    def search_and_extract(self, query: str, fetch_content: bool = False) -> List[Dict[str, Any]]:
        """
        Search and optionally extract full content from results
        
        Args:
            query: Search query
            fetch_content: Whether to fetch full page content
            
        Returns:
            Enhanced search results with optional full content
        """
        results = self.search(query)
        
        if fetch_content:
            for result in results:
                url = result.get('url')
                if url:
                    content = self.fetch_page_content(url)
                    if content:
                        result['full_content'] = content
                    time.sleep(1)  # Be respectful with requests
        
        return results
    
    def format_results_for_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a context string for LLM
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No web search results found."
        
        context = "Web Search Results:\n\n"
        
        for i, result in enumerate(results, 1):
            context += f"[Source {i}] {result['title']}\n"
            context += f"URL: {result['url']}\n"
            
            # Use full content if available, otherwise use snippet
            content = result.get('full_content', result.get('snippet', ''))
            context += f"Content: {content}\n\n"
        
        return context


class HybridSearchIntegration:
    """
    Combines web search with local RAG for hybrid retrieval
    """
    
    def __init__(self, web_search: WebSearchIntegration):
        """
        Initialize hybrid search
        
        Args:
            web_search: WebSearchIntegration instance
        """
        self.web_search = web_search
    
    def hybrid_retrieve(
        self, 
        query: str, 
        local_results: List[Dict[str, Any]], 
        web_results_count: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hybrid retrieval combining local and web results
        
        Args:
            query: User query
            local_results: Results from local RAG
            web_results_count: Number of web results to fetch
            
        Returns:
            Combined results dictionary
        """
        # Get web results
        web_results = self.web_search.search(query, max_results=web_results_count)
        
        return {
            'local_results': local_results,
            'web_results': web_results,
            'query': query,
            'retrieval_type': 'hybrid'
        }
    
    def format_hybrid_context(self, hybrid_results: Dict[str, Any]) -> str:
        """
        Format hybrid results for LLM context
        
        Args:
            hybrid_results: Dictionary with local and web results
            
        Returns:
            Formatted context string
        """
        context = "=== HYBRID RETRIEVAL RESULTS ===\n\n"
        
        # Add local results
        context += "📄 LOCAL DOCUMENTS:\n"
        local_results = hybrid_results.get('local_results', [])
        if local_results:
            for i, result in enumerate(local_results, 1):
                content = result.get('content', result.get('page_content', ''))
                source = result.get('source', 'Unknown')
                context += f"[Local {i}] Source: {source}\n"
                context += f"Content: {content[:500]}...\n\n"
        else:
            context += "No local documents found.\n\n"
        
        # Add web results
        context += "🌐 WEB SEARCH RESULTS:\n"
        web_results = hybrid_results.get('web_results', [])
        if web_results:
            context += self.web_search.format_results_for_context(web_results)
        else:
            context += "No web results found.\n\n"
        
        return context


# Test function
if __name__ == "__main__":
    # Example usage
    web_search = WebSearchIntegration(max_results=3)
    
    # Test search
    print("Testing web search...")
    results = web_search.search("latest developments in RAG systems")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
    
    # Test formatting
    print("\n\n" + "="*50)
    print("Formatted Context:")
    print("="*50)
    print(web_search.format_results_for_context(results)[:500] + "...")
