"""
Enhanced FastAPI backend for SopAssist AI with PDF processing and free LLM support
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
from pathlib import Path
import sys
import os
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import API_CONFIG, DEFAULT_LLM_PROVIDER
from core.pdf_processor import PDFProcessor
from core.vector_store import VectorStore, DocumentProcessor
from core.free_llm_interface import FreeLLMInterface, PolicyAssistant
from agents.enhanced_policy_agents import EnhancedPolicyAgents


# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None
    llm_provider: Optional[str] = DEFAULT_LLM_PROVIDER
    model: Optional[str] = None

class PolicyAnalysisRequest(BaseModel):
    question: str
    policy_text: str
    llm_provider: Optional[str] = DEFAULT_LLM_PROVIDER
    model: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    category_filter: Optional[str] = None

class UploadResponse(BaseModel):
    filename: str
    status: str
    message: str
    document_id: Optional[str] = None

class SystemStats(BaseModel):
    total_documents: int
    total_categories: int
    llm_providers: List[str]
    vector_store_stats: Dict[str, Any]
    policy_categories: Dict[str, int]


class EnhancedSopAssistAPI:
    """Enhanced API for SopAssist AI with PDF processing and free LLM support."""
    
    def __init__(self):
        """Initialize the enhanced API."""
        self.app = FastAPI(
            title="SopAssist AI - Enhanced Policy Assistant",
            description="AI-powered policy assistant with PDF processing and free LLM support",
            version="2.0.0"
        )
        
        # Initialize components
        self._initialize_components()
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing enhanced API components...")
            
            # Initialize vector store
            self.vector_store = VectorStore("policy_documents")
            self.document_processor = DocumentProcessor(self.vector_store)
            
            # Initialize PDF processor
            self.pdf_processor = PDFProcessor()
            
            # Initialize LLM interfaces
            self.llm_interfaces = {}
            self.policy_assistants = {}
            
            # Try to initialize different LLM providers
            providers_to_try = ["groq", "huggingface", "ollama"]
            
            for provider in providers_to_try:
                try:
                    if provider == "ollama":
                        from core.llm_interface import OllamaInterface
                        llm_interface = OllamaInterface()
                    else:
                        llm_interface = FreeLLMInterface(provider)
                    
                    self.llm_interfaces[provider] = llm_interface
                    self.policy_assistants[provider] = PolicyAssistant(llm_interface)
                    
                    logger.info(f"Successfully initialized {provider} LLM interface")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider} LLM interface: {e}")
            
            if not self.llm_interfaces:
                raise Exception("No LLM providers could be initialized")
            
            # Initialize enhanced policy agents
            default_provider = list(self.llm_interfaces.keys())[0]
            self.enhanced_agents = EnhancedPolicyAgents(
                self.vector_store, 
                llm_provider=default_provider
            )
            
            logger.info("✅ Enhanced API components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing API components: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "SopAssist AI - Enhanced Policy Assistant",
                "version": "2.0.0",
                "features": [
                    "PDF Policy Processing",
                    "Free LLM Support (Groq, Hugging Face, Ollama)",
                    "Agentic RAG with CrewAI",
                    "Policy Categorization",
                    "Vector Search",
                    "Policy Analysis"
                ],
                "available_llm_providers": list(self.llm_interfaces.keys())
            }
        
        @self.app.post("/ask", response_model=Dict[str, Any])
        async def ask_question(request: QuestionRequest):
            """Ask a question about policies using agentic RAG."""
            try:
                logger.info(f"Processing question: {request.question}")
                
                # Use enhanced agents for comprehensive answer
                result = self.enhanced_agents.answer_policy_question(request.question)
                
                return {
                    "question": request.question,
                    "answer": result["answer"],
                    "agents_used": result.get("agents_used", []),
                    "llm_provider": result.get("llm_provider", "unknown"),
                    "status": result.get("status", "success"),
                    "timestamp": "2024-01-01T00:00:00Z"  # Add actual timestamp
                }
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analyze-policy", response_model=Dict[str, Any])
        async def analyze_policy(request: PolicyAnalysisRequest):
            """Analyze a specific policy document."""
            try:
                provider = request.llm_provider or DEFAULT_LLM_PROVIDER
                
                if provider not in self.policy_assistants:
                    raise HTTPException(status_code=400, detail=f"LLM provider {provider} not available")
                
                assistant = self.policy_assistants[provider]
                
                # Analyze the policy
                summary = assistant.summarize_policy(request.policy_text)
                key_points = assistant.extract_key_points(request.policy_text)
                
                return {
                    "question": request.question,
                    "summary": summary,
                    "key_points": key_points,
                    "llm_provider": provider,
                    "analysis_type": "policy_analysis"
                }
                
            except Exception as e:
                logger.error(f"Error analyzing policy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/get-recommendations", response_model=Dict[str, Any])
        async def get_recommendations(request: QuestionRequest):
            """Get policy recommendations based on a question."""
            try:
                logger.info(f"Getting recommendations for: {request.question}")
                
                result = self.enhanced_agents.get_policy_recommendations(
                    request.question, 
                    request.context
                )
                
                return {
                    "question": request.question,
                    "recommendations": result["recommendations"],
                    "agents_used": result.get("agents_used", []),
                    "status": result.get("status", "success")
                }
                
            except Exception as e:
                logger.error(f"Error getting recommendations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/search-policies", response_model=List[Dict[str, Any]])
        async def search_policies(request: SearchRequest):
            """Search for relevant policy documents."""
            try:
                logger.info(f"Searching policies: {request.query}")
                
                # Perform search
                results = self.vector_store.search(
                    request.query, 
                    top_k=request.top_k
                )
                
                # Filter by category if specified
                if request.category_filter:
                    filtered_results = []
                    for result in results:
                        metadata = result.get('metadata', {})
                        categories = metadata.get('categories', [])
                        if request.category_filter.lower() in [cat.lower() for cat in categories]:
                            filtered_results.append(result)
                    results = filtered_results
                
                return results
                
            except Exception as e:
                logger.error(f"Error searching policies: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/upload-pdf", response_model=UploadResponse)
        async def upload_pdf(file: UploadFile = File(...)):
            """Upload and process a PDF policy file."""
            try:
                if not file.filename.lower().endswith('.pdf'):
                    raise HTTPException(status_code=400, detail="Only PDF files are supported")
                
                logger.info(f"Processing uploaded PDF: {file.filename}")
                
                # Save uploaded file temporarily
                temp_path = Path(f"/tmp/{file.filename}")
                with open(temp_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Process the PDF
                text = self.pdf_processor.extract_text_from_pdf(temp_path)
                
                if not text.strip():
                    raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
                
                # Categorize the policy
                categorization = self.pdf_processor.categorize_policy(file.filename)
                
                # Create document ID and metadata
                doc_id = f"uploaded_{file.filename.replace('.pdf', '')}"
                metadata = {
                    "source": "uploaded_pdf",
                    "original_file": file.filename,
                    "text_length": len(text),
                    "file_size": len(content),
                    "processing_method": "pdf_extraction",
                    "categories": categorization["categories"],
                    "year": categorization["year"],
                    "policy_type": categorization["policy_type"]
                }
                
                # Add to vector store
                self.document_processor.add_document(text, doc_id, metadata)
                
                # Clean up temp file
                temp_path.unlink()
                
                return UploadResponse(
                    filename=file.filename,
                    status="success",
                    message="PDF processed and added to knowledge base",
                    document_id=doc_id
                )
                
            except Exception as e:
                logger.error(f"Error processing uploaded PDF: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats", response_model=SystemStats)
        async def get_system_stats():
            """Get system statistics."""
            try:
                # Get vector store stats
                vector_stats = self.vector_store.get_collection_stats()
                
                # Count documents by category
                category_counts = {}
                try:
                    all_docs = self.vector_store.search("", top_k=1000)  # Get all docs
                    for doc in all_docs:
                        metadata = doc.get('metadata', {})
                        categories = metadata.get('categories', ['unknown'])
                        for category in categories:
                            category_counts[category] = category_counts.get(category, 0) + 1
                except:
                    category_counts = {}
                
                return SystemStats(
                    total_documents=vector_stats.get('total_documents', 0),
                    total_categories=len(category_counts),
                    llm_providers=list(self.llm_interfaces.keys()),
                    vector_store_stats=vector_stats,
                    policy_categories=category_counts
                )
                
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/documents/{document_id}")
        async def delete_document(document_id: str):
            """Delete a document from the vector store."""
            try:
                # Note: ChromaDB doesn't have a direct delete method in the current implementation
                # This would need to be implemented in the VectorStore class
                logger.warning("Document deletion not yet implemented")
                return {"message": "Document deletion not yet implemented", "document_id": document_id}
                
            except Exception as e:
                logger.error(f"Error deleting document: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                # Check vector store
                vector_stats = self.vector_store.get_collection_stats()
                
                # Check LLM providers
                llm_status = {}
                for provider, interface in self.llm_interfaces.items():
                    try:
                        # Simple test query
                        if hasattr(interface, 'generate_response'):
                            test_response = interface.generate_response("test", max_tokens=10)
                            llm_status[provider] = "healthy"
                        else:
                            llm_status[provider] = "unknown"
                    except:
                        llm_status[provider] = "unhealthy"
                
                return {
                    "status": "healthy",
                    "vector_store": "healthy" if vector_stats else "unhealthy",
                    "llm_providers": llm_status,
                    "timestamp": "2024-01-01T00:00:00Z"  # Add actual timestamp
                }
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "error": str(e)}
                )
    
    def run(self, host: str = None, port: int = None):
        """Run the API server."""
        host = host or API_CONFIG["host"]
        port = port or API_CONFIG["port"]
        
        logger.info(f"Starting enhanced SopAssist API on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=API_CONFIG["reload"]
        )


def main():
    """Main function to run the enhanced API."""
    try:
        api = EnhancedSopAssistAPI()
        api.run()
    except Exception as e:
        logger.error(f"Failed to start enhanced API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 