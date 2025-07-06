import streamlit as st
import requests
import json
from typing import Dict, Any
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# Configure Streamlit page
st.set_page_config(
    page_title="HR Knowledge Assistant",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_BASE_URL = "http://localhost:8000"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(file, document_type: str) -> Dict[str, Any]:
    """Upload document to backend"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"document_type": document_type}
        
        response = requests.post(
            f"{API_BASE_URL}/upload-document",
            files=files,
            data=data,
            timeout=60
        )
        
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def query_knowledge_base(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Query the knowledge base"""
    try:
        payload = {
            "query": query,
            "max_results": max_results
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30
        )
        
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_knowledge_base_stats() -> Dict[str, Any]:
    """Get knowledge base statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_documents() -> Dict[str, Any]:
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    # Header
    st.title("👥 HR Knowledge Assistant")
    st.markdown("*Ask questions about company policies, benefits, and procedures*")
    
    # Check backend connection
    if not check_backend_health():
        st.error("⚠️ Backend server is not running. Please start the FastAPI server first.")
        st.code("python -m uvicorn backend.main:app --reload", language="bash")
        return
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Document upload section
        st.subheader("Upload HR Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Upload HR policies, handbooks, or other documents"
        )
        
        document_type = st.selectbox(
            "Document Type",
            ["policy", "benefits", "leave", "conduct", "compensation", "onboarding", "general"],
            help="Categorize your document for better search results"
        )
        
        if uploaded_file and st.button("Upload Document"):
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file, document_type)
                
                if result.get("success"):
                    st.success(f"✅ {result['message']}")
                    st.info(f"📄 Processed into {result['total_chunks']} chunks")
                else:
                    st.error(f"❌ {result.get('message', 'Upload failed')}")
        
        # Knowledge base statistics
        st.subheader("📊 Knowledge Base Stats")
        if st.button("Refresh Stats"):
            stats = get_knowledge_base_stats()
            if stats.get("success"):
                vector_info = stats.get("vector_store_info", {})
                st.metric("Total Documents", stats.get("total_files", 0))
                st.metric("Vector Store Documents", vector_info.get("document_count", 0))
            else:
                st.error("Failed to load stats")
        
        # Document list
        st.subheader("📋 Uploaded Documents")
        if st.button("Show Documents"):
            docs = get_documents()
            if docs.get("success"):
                for doc in docs.get("documents", []):
                    st.text(f"📄 {doc.get('filename', 'Unknown')}")
                    st.caption(f"Size: {doc.get('file_size', 0) / 1024:.1f} KB")
            else:
                st.error("Failed to load documents")
    
    # Main chat interface
    st.header("💬 Ask HR Questions")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.text(f"📄 {source['filename']} (Score: {source['relevance_score']:.3f})")
    
    # Chat input
    if prompt := st.chat_input("Ask about HR policies, benefits, leave, etc."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = query_knowledge_base(prompt)
                
                if result.get("success"):
                    response = result.get("answer", "No answer available")
                    st.markdown(response)
                    
                    # Show category and sources
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"🏷️ Category: {result.get('category', 'Unknown')}")
                    with col2:
                        st.caption(f"📊 Relevant chunks: {result.get('relevant_chunks', 0)}")
                    
                    # Show sources
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.text(f"📄 {source['filename']} (Score: {source['relevance_score']:.3f})")
                                if source.get('page') != 'N/A':
                                    st.caption(f"Page: {source['page']}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                else:
                    error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Example queries
    st.subheader("💡 Example Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Benefits & Leave:**
        - How many vacation days do I get?
        - What's covered by health insurance?
        - How do I request parental leave?
        """)
    
    with col2:
        st.markdown("""
        **Policies & Procedures:**
        - What's the remote work policy?
        - How do I report harassment?
        - What's the dress code?
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*HR Knowledge Assistant - Powered by AI*")

if __name__ == "__main__":
    main() 