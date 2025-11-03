"""
Intelligent Source Router
Main routing logic that coordinates between local RAG and web search
"""

from typing import Dict, Any, List, Optional
from query_classifier import QueryClassifier
from web_search_integration import WebSearchIntegration, HybridSearchIntegration
import os


class IntelligentSourceRouter:
    """
    Intelligent router that determines and executes the optimal data source strategy
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        enable_web_search: bool = True,
        web_max_results: int = 5
    ):
        """
        Initialize the intelligent source router
        
        Args:
            api_key: OpenAI API key
            enable_web_search: Whether to enable web search functionality
            web_max_results: Maximum web search results to retrieve
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize components
        self.query_classifier = QueryClassifier(api_key=self.api_key)
        self.web_search = WebSearchIntegration(max_results=web_max_results) if enable_web_search else None
        self.hybrid_search = HybridSearchIntegration(self.web_search) if enable_web_search else None
        
        self.routing_history = []
    
    def route_query(
        self, 
        query: str, 
        local_retriever=None,
        has_uploaded_docs: bool = False
    ) -> Dict[str, Any]:
        """
        Route a query to the appropriate data source(s)
        
        Args:
            query: User's query string
            local_retriever: Local RAG retriever instance (FAISS, etc.)
            has_uploaded_docs: Whether user has uploaded documents
            
        Returns:
            Dictionary containing routing decision and retrieved context
        """
        # Step 1: Classify the query
        classification = self.query_classifier.classify_query(query, has_uploaded_docs)
        
        # Step 2: Execute retrieval based on classification
        datasource = classification['datasource']
        
        result = {
            'query': query,
            'routing': classification,
            'context': '',
            'sources': []
        }
        
        try:
            if datasource == 'local_rag':
                result.update(self._retrieve_local(query, local_retriever))
                
            elif datasource == 'web_search':
                result.update(self._retrieve_web(query))
                
            elif datasource == 'hybrid':
                result.update(self._retrieve_hybrid(query, local_retriever))
            
            # Save routing history
            self._save_routing_history(result)
            
        except Exception as e:
            print(f"Routing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _retrieve_local(self, query: str, local_retriever) -> Dict[str, Any]:
        """Retrieve from local RAG system"""
        if local_retriever is None:
            return {
                'context': 'No local documents available.',
                'sources': [],
                'retrieval_type': 'local_rag',
                'error': 'No local retriever configured'
            }
        
        try:
            # Retrieve from local vector store
            docs = local_retriever.get_relevant_documents(query)
            
            # Format context
            context = "=== LOCAL DOCUMENT RESULTS ===\n\n"
            sources = []
            
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                
                context += f"[Document {i}]\n"
                context += f"Source: {source} (Page: {page})\n"
                context += f"Content: {doc.page_content}\n\n"
                
                sources.append({
                    'type': 'local',
                    'source': source,
                    'page': page,
                    'content': doc.page_content
                })
            
            return {
                'context': context,
                'sources': sources,
                'retrieval_type': 'local_rag',
                'num_results': len(docs)
            }
            
        except Exception as e:
            return {
                'context': f'Error retrieving local documents: {str(e)}',
                'sources': [],
                'retrieval_type': 'local_rag',
                'error': str(e)
            }
    
    def _retrieve_web(self, query: str) -> Dict[str, Any]:
        """Retrieve from web search"""
        if self.web_search is None:
            return {
                'context': 'Web search is not enabled.',
                'sources': [],
                'retrieval_type': 'web_search',
                'error': 'Web search not enabled'
            }
        
        try:
            # Perform web search
            web_results = self.web_search.search(query)
            
            # Format context
            context = self.web_search.format_results_for_context(web_results)
            
            sources = [
                {
                    'type': 'web',
                    'title': r['title'],
                    'url': r['url'],
                    'snippet': r['snippet']
                }
                for r in web_results
            ]
            
            return {
                'context': context,
                'sources': sources,
                'retrieval_type': 'web_search',
                'num_results': len(web_results)
            }
            
        except Exception as e:
            return {
                'context': f'Error performing web search: {str(e)}',
                'sources': [],
                'retrieval_type': 'web_search',
                'error': str(e)
            }
    
    def _retrieve_hybrid(self, query: str, local_retriever) -> Dict[str, Any]:
        """Retrieve from both local and web sources"""
        if self.hybrid_search is None:
            # Fallback to local only
            return self._retrieve_local(query, local_retriever)
        
        try:
            # Get local results
            local_docs = []
            if local_retriever:
                local_docs = local_retriever.get_relevant_documents(query)
            
            local_results = [
                {
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'content': doc.page_content
                }
                for doc in local_docs
            ]
            
            # Perform hybrid retrieval
            hybrid_results = self.hybrid_search.hybrid_retrieve(
                query=query,
                local_results=local_results,
                web_results_count=3
            )
            
            # Format context
            context = self.hybrid_search.format_hybrid_context(hybrid_results)
            
            # Combine sources
            sources = [
                {'type': 'local', **lr} for lr in local_results
            ] + [
                {'type': 'web', **wr} for wr in hybrid_results['web_results']
            ]
            
            return {
                'context': context,
                'sources': sources,
                'retrieval_type': 'hybrid',
                'num_local_results': len(local_results),
                'num_web_results': len(hybrid_results['web_results'])
            }
            
        except Exception as e:
            return {
                'context': f'Error in hybrid retrieval: {str(e)}',
                'sources': [],
                'retrieval_type': 'hybrid',
                'error': str(e)
            }
    
    def _save_routing_history(self, result: Dict[str, Any]):
        """Save routing decision to history"""
        self.routing_history.append({
            'query': result['query'],
            'datasource': result['routing']['datasource'],
            'reasoning': result['routing']['reasoning'],
            'confidence': result['routing']['confidence'],
            'num_sources': len(result.get('sources', []))
        })
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions"""
        if not self.routing_history:
            return {
                'total_queries': 0,
                'by_source': {},
                'avg_confidence': 0.0
            }
        
        total = len(self.routing_history)
        by_source = {}
        total_confidence = 0.0
        
        for entry in self.routing_history:
            source = entry['datasource']
            by_source[source] = by_source.get(source, 0) + 1
            total_confidence += entry['confidence']
        
        return {
            'total_queries': total,
            'by_source': by_source,
            'avg_confidence': total_confidence / total if total > 0 else 0.0,
            'routing_history': self.routing_history
        }


# Test function
if __name__ == "__main__":
    # Example usage
    router = IntelligentSourceRouter()
    
    # Test queries
    test_queries = [
        "What does my uploaded PDF say about machine learning?",
        "What are the latest AI news today?",
        "Compare information from my documents with current trends"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = router.route_query(query, has_uploaded_docs=True)
        
        print(f"Routed to: {result['routing']['datasource']}")
        print(f"Reasoning: {result['routing']['reasoning']}")
        print(f"Confidence: {result['routing']['confidence']:.2f}")
        print(f"Sources found: {len(result.get('sources', []))}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("ROUTING STATISTICS")
    print('='*60)
    stats = router.get_routing_stats()
    print(f"Total queries: {stats['total_queries']}")
    print(f"By source: {stats['by_source']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
