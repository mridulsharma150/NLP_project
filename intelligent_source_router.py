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
            
            # Use enhanced search with Tavily as primary
            web_results = self.web_search.search_and_extract(
                query,
                fetch_content=self.fetch_full_content
            )
            
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
                
                content_to_use = full_content if full_content else snippet
                
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
                'num_results': len(docs)
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