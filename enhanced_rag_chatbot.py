"""
Enhanced RAG Chatbot with Intelligent Response Generation
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from PIL import Image
import easyocr
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

# Import routing components
from intelligent_source_router import IntelligentSourceRouter
from query_classifier import QueryClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enhanced NLP RAG Chatbot - With Live Web Search",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'source_router' not in st.session_state:
    st.session_state.source_router = None
if 'uploaded_files_processed' not in st.session_state:
    st.session_state.uploaded_files_processed = []


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num + 1}]\n{page_text}"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        logger.error(f"PDF extraction error: {e}")
        return ""


def extract_text_from_image(image_file):
    """Extract text from image using EasyOCR"""
    try:
        reader = easyocr.Reader(['en'])
        image = Image.open(image_file)
        results = reader.readtext(image)
        text = "\n".join([result[1] for result in results])
        return text
    except Exception as e:
        st.error(f"Error extracting image text: {e}")
        logger.error(f"Image extraction error: {e}")
        return ""


def process_uploaded_files(uploaded_files):
    """Process uploaded files and create/update vector store"""
    if not uploaded_files:
        return None

    with st.spinner("Processing uploaded files..."):
        all_texts = []

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            if file_name in st.session_state.uploaded_files_processed:
                continue

            try:
                if file_name.endswith('.pdf'):
                    text = extract_text_from_pdf(uploaded_file)
                elif file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    text = extract_text_from_image(uploaded_file)
                else:
                    st.warning(f"Unsupported file type: {file_name}")
                    continue

                if text:
                    all_texts.append({
                        'content': text,
                        'source': file_name
                    })
                    st.session_state.uploaded_files_processed.append(file_name)

            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
                logger.error(f"File processing error for {file_name}: {e}")

        if not all_texts:
            return None

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        documents = []
        for item in all_texts:
            chunks = text_splitter.split_text(item['content'])
            for i, chunk in enumerate(chunks):
                documents.append({
                    'page_content': chunk,
                    'metadata': {
                        'source': item['source'],
                        'chunk': i
                    }
                })

        # Create embeddings
        try:
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
            logger.error(f"Embeddings error: {e}")
            return None

        # Create or update vector store
        texts = [doc['page_content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]

        try:
            if st.session_state.vectorstore is None:
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            else:
                vectorstore = st.session_state.vectorstore
                vectorstore.add_texts(texts, metadatas=metadatas)

            st.success(f"✅ Processed {len(all_texts)} file(s) into {len(documents)} chunks")
            logger.info(f"Processed {len(all_texts)} files into {len(documents)} chunks")
            return vectorstore

        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            logger.error(f"Vector store error: {e}")
            return None


def initialize_source_router():
    """Initialize the intelligent source router"""
    if st.session_state.source_router is None:
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            st.session_state.source_router = IntelligentSourceRouter(
                api_key=api_key,
                enable_web_search=True,
                web_max_results=5,
                fetch_full_content=True
            )
            logger.info("Source router initialized with full content fetching enabled")
        except Exception as e:
            st.error(f"Error initializing source router: {e}")
            logger.error(f"Source router initialization error: {e}")


def is_self_contained_query(query: str) -> bool:
    """
    Check if query can be answered using LLM's own knowledge
    These queries don't need web search context
    
    Returns True if LLM can answer directly
    """
    query_lower = query.lower()
    
    # Math and logic questions
    math_indicators = [
        'calculate', 'compute', 'multiply', 'divide', 'add', 'subtract',
        'what is', 'equals', '=', 'plus', 'minus', 'times',
        'is', 'are', 'can', 'could', 'would'
    ]
    
    # Self-contained query indicators
    self_contained = [
        # Basic math
        r'\d+\s*[\+\-\*/]\s*\d+',  # "15 * 24"
        r'what is \d+',  # "what is 15"
        
        # General knowledge that doesn't change
        'capital of',
        'definition of',
        'how to',
        'what does .* mean',
        'synonym for',
        'opposite of',
        'formula for',
        'difference between',
        'similarity between'
    ]
    
    import re
    
    # Check for mathematical expressions
    if any(re.search(pattern, query_lower) for pattern in self_contained):
        return True
    
    # Check for math indicators with numbers
    has_numbers = any(char.isdigit() for char in query)
    if has_numbers and any(ind in query_lower for ind in math_indicators):
        return True
    
    return False


def is_factual_query(query: str) -> bool:
    """
    Check if query requires current/factual web information
    Returns True if web search is strongly recommended
    """
    query_lower = query.lower()
    
    factual_indicators = [
        'latest', 'current', 'recent', 'today', 'this week', 'this month',
        'news', 'breaking', 'happening',
        'weather', 'temperature', 'forecast',
        '2025', '2024', 'this year',
        'how is', 'what is happening',
        'status of', 'update on'
    ]
    
    return any(ind in query_lower for ind in factual_indicators)


def generate_response_with_context(prompt: str, routing_result: Dict) -> str:
    """
    IMPROVED: Generate response with smart context usage
    - Uses web search context when necessary
    - Allows LLM knowledge for self-contained questions
    - Avoids redundant preambles
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Get routing info
        datasource = routing_result['routing']['datasource']
        context = routing_result.get('context', '')
        retrieval_type = routing_result.get('retrieval_type', 'unknown')
        
        # IMPROVED LOGIC: Check if this is a self-contained query
        is_self_contained = is_self_contained_query(prompt)
        is_factual = is_factual_query(prompt)
        
        logger.info(f"Query analysis: self_contained={is_self_contained}, factual={is_factual}")
        
        # Determine which system prompt to use
        if is_self_contained and not is_factual:
            # SELF-CONTAINED QUERY: Allow LLM to use own knowledge
            # No forced web search context
            logger.info("Using LLM knowledge (self-contained query)")
            
            system_prompt = f"""You are a helpful, knowledgeable assistant.

For this query, provide a clear, direct answer.
- If it's a math problem, solve it step-by-step
- If it's a definition or explanation, provide clear information
- If it's logic/reasoning, use your knowledge

Be concise and direct. No need to cite sources for fundamental knowledge."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {prompt}"}
            ]
        
        elif datasource == 'web_search' or retrieval_type == 'web_search':
            # WEB SEARCH QUERY: Use the fetched context
            logger.info("Using web search context")
            
            current_date = datetime.now().strftime('%B %d, %Y')
            
            system_prompt = f"""You are a helpful assistant with access to current web search results.

Based on the search results provided below, answer the user's question.

CURRENT DATE: {current_date}

WEB SEARCH RESULTS:
{context}

Rules:
- Answer based primarily on the search results provided
- If search results don't contain information, you may use general knowledge
- Cite sources when using specific information from search results
- Be clear and concise"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {prompt}"}
            ]
        
        elif datasource == 'local_rag':
            # LOCAL DOCUMENT QUERY: Use document context
            logger.info("Using local document context")
            
            system_prompt = f"""You are a helpful assistant with access to uploaded documents.

Based on the document content provided below, answer the user's question.

DOCUMENT CONTENT:
{context}

Rules:
- Answer based on the document content
- If information is not in the documents, clearly state that
- Cite which document you're referencing
- Be clear and concise"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {prompt}"}
            ]
        
        else:  # hybrid
            # HYBRID QUERY: Combine sources
            logger.info("Using hybrid context (local + web)")
            
            system_prompt = f"""You are a helpful assistant with access to both uploaded documents and web search results.

Use the provided sources to answer comprehensively:

SOURCES:
{context}

Rules:
- Use both local documents and web information
- Distinguish between document content and web information
- Be clear about sources
- Provide comprehensive answer"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {prompt}"}
            ]
        
        # Generate response
        logger.info(f"Generating response for: {prompt[:50]}...")
        response = llm.invoke(messages)
        answer = response.content
        
        return answer

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"


def main():
    st.title("🤖 Enhanced NLP RAG Chatbot with Live Web Search")

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload (Optional)")
        st.info("Upload PDFs or images to enable document search")

        uploaded_files = st.file_uploader(
            "Upload PDFs or Images",
            type=['pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("🔄 Process Files"):
                vectorstore = process_uploaded_files(uploaded_files)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.rerun()

        st.divider()

        # Display processed files
        if st.session_state.uploaded_files_processed:
            st.subheader("✅ Processed Files")
            for file_name in st.session_state.uploaded_files_processed:
                st.text(f"📄 {file_name}")

        st.divider()

        # Routing statistics
        if st.session_state.source_router:
            stats = st.session_state.source_router.get_routing_stats()
            if stats['total_queries'] > 0:
                st.subheader("📊 Routing Stats")
                st.metric("Total Queries", stats['total_queries'])

                col1, col2 = st.columns(2)
                with col1:
                    for source, count in stats['by_source'].items():
                        st.metric(source.replace('_', ' ').title(), count)
                with col2:
                    st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
                    st.metric("Success Rate", f"{stats['success_rate']:.1%}")

        st.divider()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Initialize router
    initialize_source_router()

    # Chat interface
    st.header("💬 Ask Anything")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display routing info if available
            if "routing" in message:
                with st.expander("🔍 Source Information"):
                    routing = message["routing"]
                    source_name = routing['datasource'].replace('_', ' ').title()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Source", source_name)
                    with col2:
                        st.metric("Confidence", f"{routing['confidence']:.1%}")
                    with col3:
                        st.metric("Type", message.get('retrieval_type', 'unknown').replace('_', ' ').title())

                    st.info(f"**Reasoning**: {routing['reasoning']}")

            # Display sources
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 {len(message['sources'])} Sources Used"):
                    for i, source in enumerate(message["sources"][:10], 1):
                        if source.get('type') == 'web':
                            st.markdown(f"**{i}. 🌐 {source.get('title', 'Web Source')}**")
                            if source.get('url'):
                                st.caption(f"URL: {source['url']}")
                        else:
                            st.markdown(f"**{i}. 📄 {source.get('source', 'Local Document')}**")

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get retriever if available
                    retriever = None
                    has_docs = st.session_state.vectorstore is not None

                    if has_docs:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

                    # Route the query
                    routing_result = st.session_state.source_router.route_query(
                        query=prompt,
                        local_retriever=retriever,
                        has_uploaded_docs=has_docs
                    )

                    logger.info(f"Query routed to: {routing_result['routing']['datasource']}")

                    # Generate response with improved logic
                    answer = generate_response_with_context(prompt, routing_result)

                    st.markdown(answer)

                    # Show routing details
                    with st.expander("🔍 Full Routing Details"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Routing Decision")
                            routing = routing_result['routing']
                            st.write(f"**Data Source**: {routing['datasource'].replace('_', ' ').title()}")
                            st.write(f"**Reasoning**: {routing['reasoning']}")
                            st.write(f"**Confidence**: {routing['confidence']:.1%}")

                        with col2:
                            st.subheader("Search Results")
                            retrieval_type = routing_result.get('retrieval_type', 'unknown')
                            st.write(f"**Retrieval Type**: {retrieval_type.replace('_', ' ').title()}")
                            st.write(f"**Results Found**: {len(routing_result.get('sources', []))}")

                            if routing_result.get('num_web_results'):
                                st.write(f"**Web Results**: {routing_result['num_web_results']}")
                            if routing_result.get('num_local_results'):
                                st.write(f"**Local Results**: {routing_result['num_local_results']}")

                        # Show full context preview
                        st.subheader("Retrieved Context (Preview)")
                        context_preview = routing_result.get('context', '')[:1000]
                        st.text_area("Context:", value=context_preview + ("..." if len(routing_result.get('context', '')) > 1000 else ""), height=200, disabled=True)

                    # Save assistant message with routing info
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "routing": routing_result['routing'],
                        "sources": routing_result.get('sources', []),
                        "retrieval_type": routing_result.get('retrieval_type', 'unknown')
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Response generation error: {e}", exc_info=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "routing": {"datasource": "error", "reasoning": str(e), "confidence": 0},
                        "sources": []
                    })


if __name__ == "__main__":
    main()
