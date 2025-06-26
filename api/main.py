"""
FastAPI main application for SopAssist AI
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
from pathlib import Path
import sys
import json
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import API_CONFIG, LOGGING_CONFIG
from core.ocr_processor import OCRProcessor
from core.vector_store import VectorStore, DocumentProcessor
from core.llm_interface import OllamaInterface, PolicyAssistant
from agents.policy_agents import PolicyAgents
from core.policy_loader import PolicyLoader

# Configure logging
logger.add(
    LOGGING_CONFIG["file"],
    format=LOGGING_CONFIG["format"],
    level=LOGGING_CONFIG["level"],
    rotation="10 MB"
)

logger.add(
    LOGGING_CONFIG["error_file"],
    format=LOGGING_CONFIG["format"],
    level="ERROR",
    rotation="10 MB"
)

# Initialize FastAPI app
app = FastAPI(
    title="SopAssist AI API",
    description="AI-powered policy assistant for company SOP/HR/tech policies",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    use_agents: bool = True

class PolicyAnalysisRequest(BaseModel):
    document_text: str

class RecommendationsRequest(BaseModel):
    question: str
    context: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    files_processed: int
    errors: List[str] = []

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# Global instances (initialize on startup)
ocr_processor = None
vector_store = None
document_processor = None
llm_interface = None
policy_assistant = None
policy_agents = None
policy_loader = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global ocr_processor, vector_store, document_processor, llm_interface, policy_assistant, policy_agents, policy_loader
    
    try:
        logger.info("Initializing SopAssist AI components...")
        
        # Initialize OCR processor
        ocr_processor = OCRProcessor()
        logger.info("✓ OCR processor initialized")
        
        # Initialize vector store
        vector_store = VectorStore()
        logger.info("✓ Vector store initialized")
        
        # Initialize document processor
        document_processor = DocumentProcessor(vector_store)
        logger.info("✓ Document processor initialized")
        
        # Initialize LLM interface (for direct LLM calls)
        llm_interface = OllamaInterface()
        logger.info("✓ LLM interface initialized")
        
        # Initialize policy assistant
        policy_assistant = PolicyAssistant(llm_interface)
        logger.info("✓ Policy assistant initialized")
        
        # Initialize policy agents with ChatOllama-compatible signature
        policy_agents = PolicyAgents(vector_store, model="llama3", base_url="http://localhost:11434")
        logger.info("✓ Policy agents initialized")
        
        # Initialize policy loader
        policy_loader = PolicyLoader()
        logger.info("✓ Policy loader initialized")
        
        logger.info("✅ All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SopAssist AI API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if all components are initialized
        if all([ocr_processor, vector_store, llm_interface, policy_assistant, policy_loader]):
            policy_summary = policy_loader.get_policy_summary()
            return {
                "status": "healthy",
                "components": {
                    "ocr_processor": "initialized",
                    "vector_store": "initialized",
                    "llm_interface": "initialized",
                    "policy_assistant": "initialized",
                    "policy_agents": "initialized" if policy_agents else "not_initialized",
                    "policy_loader": "initialized"
                },
                "policies_loaded": policy_summary['total_policies']
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Some components are not initialized"
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about policies."""
    try:
        if not policy_assistant or not policy_loader:
            raise HTTPException(status_code=503, detail="Policy assistant or loader not initialized")
        
        # Get relevant policy content for the question
        policy_context = policy_loader.get_policy_content_for_llm(request.question)
        
        if request.use_agents and policy_agents:
            # Use CrewAI agents for complex reasoning
            result = policy_agents.answer_question(request.question)
            return {
                "answer": result["answer"],
                "question": request.question,
                "method": "agents",
                "status": result["status"],
                "policies_used": len(policy_loader.get_all_policies())
            }
        else:
            # Use simple LLM with policy context
            if not policy_context or policy_context == "No policies loaded.":
                return {
                    "answer": "I couldn't find any policy documents to answer your question. Please ensure policies have been extracted first.",
                    "question": request.question,
                    "method": "simple_llm",
                    "status": "no_policies"
                }
            
            # Generate answer using policy context
            answer = policy_assistant.answer_policy_question(request.question, [policy_context])
            
            return {
                "answer": answer,
                "question": request.question,
                "method": "simple_llm",
                "status": "success",
                "policies_used": len(policy_loader.get_all_policies())
            }
            
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/analyze")
async def analyze_policy(request: PolicyAnalysisRequest):
    """Analyze a policy document."""
    try:
        if not policy_assistant:
            raise HTTPException(status_code=503, detail="Policy assistant not initialized")
        
        # Generate summary and key points
        summary = policy_assistant.summarize_policy(request.document_text)
        key_points = policy_assistant.extract_key_points(request.document_text)
        
        return {
            "summary": summary,
            "key_points": key_points,
            "document_length": len(request.document_text),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing policy: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing policy: {str(e)}")

@app.post("/recommendations")
async def get_recommendations(request: RecommendationsRequest):
    """Get policy recommendations."""
    try:
        if not policy_agents:
            raise HTTPException(status_code=503, detail="Policy agents not initialized")
        
        result = policy_agents.get_policy_recommendations(request.question, request.context)
        
        return {
            "recommendations": result["recommendations"],
            "question": request.question,
            "context": request.context,
            "status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/search")
async def search_policies(request: SearchRequest):
    """Search for policy documents."""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        results = vector_store.search(request.query, top_k=request.top_k)
        
        return {
            "query": request.query,
            "results": [
                {
                    "content": result["document"],
                    "similarity": result["similarity"],
                    "metadata": result["metadata"]
                }
                for result in results
            ],
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching policies: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching policies: {str(e)}")

@app.post("/upload")
async def upload_policy_images(files: List[UploadFile] = File(...)):
    """Upload and process policy images."""
    try:
        if not ocr_processor or not document_processor:
            raise HTTPException(status_code=503, detail="OCR processor not initialized")
        
        from config.settings import IMAGES_DIR, PROCESSED_DIR
        
        processed_count = 0
        errors = []
        
        for file in files:
            try:
                # Check file type
                if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                    errors.append(f"Unsupported file type: {file.filename}")
                    continue
                
                # Save file
                file_path = IMAGES_DIR / file.filename
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Extract text using OCR
                text = ocr_processor.extract_text(file_path)
                
                if text.strip():
                    # Add to vector store
                    doc_id = f"uploaded_{file.filename}_{processed_count}"
                    metadata = {
                        "source": "upload",
                        "filename": file.filename,
                        "file_type": file.content_type
                    }
                    
                    document_processor.add_document(text, doc_id, metadata)
                    processed_count += 1
                    
                    # Save extracted text
                    text_file = PROCESSED_DIR / f"{file_path.stem}.txt"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(text)
                        
                else:
                    errors.append(f"No text extracted from {file.filename}")
                    
            except Exception as e:
                errors.append(f"Error processing {file.filename}: {str(e)}")
        
        return UploadResponse(
            message=f"Successfully processed {processed_count} files",
            files_processed=processed_count,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        stats = vector_store.get_collection_stats()
        
        # Get model info
        model_info = {}
        if llm_interface:
            model_info = llm_interface.get_model_info()
        
        return {
            "vector_store": stats,
            "model": model_info,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the vector store."""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        # Find all chunks for this document
        results = vector_store.search(doc_id, top_k=100)  # Get many results to find all chunks
        chunk_ids = []
        
        for result in results:
            if result["metadata"].get("original_doc_id") == doc_id:
                # Extract chunk ID from the result
                # This is a simplified approach - in production you'd want a more robust method
                chunk_ids.append(result.get("id", ""))
        
        if chunk_ids:
            vector_store.delete_documents(chunk_ids)
            return {"message": f"Deleted document {doc_id} and {len(chunk_ids)} chunks"}
        else:
            return {"message": f"No chunks found for document {doc_id}"}
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
        workers=API_CONFIG["workers"]
    ) 