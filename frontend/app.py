"""
Streamlit Frontend for SopAssist AI
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import API_CONFIG, STREAMLIT_CONFIG

# Configure page
st.set_page_config(
    page_title="SopAssist AI",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def ask_question(question: str, use_agents: bool = True) -> Dict[str, Any]:
    """Send a question to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "use_agents": use_agents},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def search_policies(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for policy documents."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def analyze_policy(document_text: str) -> Dict[str, Any]:
    """Analyze a policy document."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={"document_text": document_text},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_recommendations(question: str, context: str = None) -> Dict[str, Any]:
    """Get policy recommendations."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommendations",
            json={"question": question, "context": context},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def upload_files(files: List) -> Dict[str, Any]:
    """Upload policy images."""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), file.type)))
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files_data,
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_stats() -> Dict[str, Any]:
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">📋 SopAssist AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-powered policy assistant</p>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not running. Please start the API server first.")
        st.info("Run: `python api/main.py`")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Ask Questions", "Search Policies", "Analyze Document", "Upload Files", "System Stats"]
    )
    
    # Health status in sidebar
    if health_data:
        st.sidebar.success("✅ API Connected")
        if "components" in health_data:
            for component, status in health_data["components"].items():
                if status == "initialized":
                    st.sidebar.success(f"✓ {component.replace('_', ' ').title()}")
                else:
                    st.sidebar.error(f"✗ {component.replace('_', ' ').title()}")
    
    # Main content based on selected page
    if page == "Ask Questions":
        show_ask_questions_page()
    elif page == "Search Policies":
        show_search_policies_page()
    elif page == "Analyze Document":
        show_analyze_document_page()
    elif page == "Upload Files":
        show_upload_files_page()
    elif page == "System Stats":
        show_system_stats_page()

def show_ask_questions_page():
    """Show the ask questions page."""
    st.header("🤔 Ask Questions About Policies")
    
    # Question input
    question = st.text_area(
        "What would you like to know about your company policies?",
        placeholder="e.g., What is the vacation policy? How do I submit an expense report?",
        height=100
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        use_agents = st.checkbox("Use AI Agents (Advanced Reasoning)", value=True)
    with col2:
        if st.button("Ask Question", type="primary"):
            if question.strip():
                with st.spinner("Thinking..."):
                    result = ask_question(question, use_agents)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display answer
                    st.success("✅ Answer Generated")
                    
                    # Method used
                    method = result.get("method", "unknown")
                    if method == "agents":
                        st.info("🤖 Used AI Agents for advanced reasoning")
                    else:
                        st.info("🔍 Used simple search and analysis")
                    
                    # Answer
                    st.markdown("### Answer:")
                    st.write(result["answer"])
                    
                    # Sources (if available)
                    if "sources" in result and result["sources"]:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.write(source["content"])
                                st.write(f"Relevance: {source['similarity']:.3f}")
                                st.divider()
            else:
                st.warning("Please enter a question.")

def show_search_policies_page():
    """Show the search policies page."""
    st.header("🔍 Search Policy Documents")
    
    # Search input
    query = st.text_input(
        "Search for policy documents:",
        placeholder="e.g., vacation, expense, security"
    )
    
    # Search options
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)
    with col2:
        if st.button("Search", type="primary"):
            if query.strip():
                with st.spinner("Searching..."):
                    result = search_policies(query, top_k)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"Found {result['total_results']} results")
                    
                    # Display results
                    for i, doc in enumerate(result["results"], 1):
                        with st.expander(f"Document {i} (Similarity: {doc['similarity']:.3f})"):
                            st.write(doc["content"])
                            if doc["metadata"]:
                                st.write("**Metadata:**", doc["metadata"])
            else:
                st.warning("Please enter a search query.")

def show_analyze_document_page():
    """Show the analyze document page."""
    st.header("📄 Analyze Policy Document")
    
    # Document input
    document_text = st.text_area(
        "Paste your policy document text here:",
        placeholder="Paste the text of a policy document to analyze...",
        height=300
    )
    
    if st.button("Analyze Document", type="primary"):
        if document_text.strip():
            with st.spinner("Analyzing..."):
                result = analyze_policy(document_text)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("✅ Analysis Complete")
                
                # Summary
                st.markdown("### 📝 Summary")
                st.write(result["summary"])
                
                # Key points
                st.markdown("### 🔑 Key Points")
                for point in result["key_points"]:
                    st.write(f"• {point}")
                
                # Document info
                st.info(f"Document length: {result['document_length']} characters")
        else:
            st.warning("Please enter document text to analyze.")

def show_upload_files_page():
    """Show the upload files page."""
    st.header("📤 Upload Policy Images")
    
    st.info("""
    Upload images of policy documents to add them to the knowledge base.
    Supported formats: JPG, JPEG, PNG, TIFF, BMP
    """)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose policy document images:",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files:")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size} bytes)")
        
        if st.button("Upload and Process", type="primary"):
            with st.spinner("Processing images..."):
                result = upload_files(uploaded_files)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"✅ {result['message']}")
                
                if result["errors"]:
                    st.warning("Some files had issues:")
                    for error in result["errors"]:
                        st.write(f"• {error}")

def show_system_stats_page():
    """Show the system statistics page."""
    st.header("📊 System Statistics")
    
    if st.button("Refresh Stats", type="primary"):
        with st.spinner("Loading statistics..."):
            stats = get_stats()
        
        if "error" in stats:
            st.error(f"Error: {stats['error']}")
        else:
            # Vector store stats
            if "vector_store" in stats:
                vs_stats = stats["vector_store"]
                st.markdown("### 🗄️ Vector Store")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", vs_stats.get("total_documents", 0))
                with col2:
                    st.metric("Collection", vs_stats.get("collection_name", "N/A"))
                with col3:
                    st.metric("Embedding Dimension", vs_stats.get("embedding_dimension", "N/A"))
            
            # Model info
            if "model" in stats:
                model_info = stats["model"]
                st.markdown("### 🤖 Model Information")
                if "error" not in model_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", model_info.get("name", "N/A"))
                    with col2:
                        st.metric("Size", model_info.get("size", "N/A"))
                    with col3:
                        st.metric("Modified", model_info.get("modified_at", "N/A")[:10] if model_info.get("modified_at") else "N/A")
                else:
                    st.error(f"Model error: {model_info['error']}")

if __name__ == "__main__":
    main() 