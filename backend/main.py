"""
FastAPI Backend for SopAssist AI with Bengali language support
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.ocr_processor import OCRProcessor
from core.vector_store import VectorStore
from core.free_llm_interface import FreeLLMInterface, PolicyAssistant
from core.pdf_processor import PDFProcessor
from config.settings import API_CONFIG, DEFAULT_LLM_PROVIDER

# Initialize FastAPI app
app = FastAPI(
    title="SopAssist AI API",
    description="AI-powered SOP and Policy Assistant with Bengali language support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger.info("Initializing SopAssist AI components...")

# Initialize OCR processor with Bengali support
ocr_processor = OCRProcessor()

# Initialize PDF processor with Bengali support
pdf_processor = PDFProcessor()

# Initialize vector store
vector_store = VectorStore()

# Initialize LLM interface
try:
    llm_interface = FreeLLMInterface(provider=DEFAULT_LLM_PROVIDER)
    policy_assistant = PolicyAssistant(llm_interface)
    logger.info(f"Initialized LLM interface with provider: {DEFAULT_LLM_PROVIDER}")
except Exception as e:
    logger.error(f"Failed to initialize LLM interface: {e}")
    llm_interface = None
    policy_assistant = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    similarity_threshold: float = 0.7

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    language: str

class ProcessFileResponse(BaseModel):
    message: str
    file_id: str
    text_extracted: str
    chunks_created: int

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    model_info: Dict[str, Any]

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all components."""
    components = {
        "ocr_processor": "✅" if ocr_processor else "❌",
        "pdf_processor": "✅" if pdf_processor else "❌",
        "vector_store": "✅" if vector_store else "❌",
        "llm_interface": "✅" if llm_interface else "❌"
    }
    
    model_info = {}
    if llm_interface:
        model_info = llm_interface.get_model_info()
    
    return HealthResponse(
        status="healthy" if all("✅" in status for status in components.values()) else "degraded",
        components=components,
        model_info=model_info
    )

# Process image file endpoint
@app.post("/process-image", response_model=ProcessFileResponse)
async def process_image(file: UploadFile = File(...)):
    """Process an image file and extract text using OCR with Bengali support."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        # Process with OCR
        text = ocr_processor.extract_text_from_image(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the image")
        
        # Store in vector database
        file_id = f"img_{file.filename}_{hash(content)}"
        chunks = vector_store.add_texts([text], [{"source": file.filename, "type": "image", "file_id": file_id}])
        
        return ProcessFileResponse(
            message="Image processed successfully",
            file_id=file_id,
            text_extracted=text,
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Process PDF file endpoint
@app.post("/process-pdf", response_model=ProcessFileResponse)
async def process_pdf(file: UploadFile = File(...)):
    """Process a PDF file and extract text with Bengali support."""
    try:
        # Validate file type
        if not file.content_type == 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        content = await file.read()
        
        # Process PDF
        result = pdf_processor.process_pdf(content, file.filename)
        
        if not result['text'] or not result['text'].strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Store in vector database
        file_id = f"pdf_{file.filename}_{hash(content)}"
        chunks = vector_store.add_texts(
            [result['text']], 
            [{
                "source": file.filename, 
                "type": "pdf", 
                "file_id": file_id,
                "category": result.get('category', 'general'),
                "language": result.get('language', 'english')
            }]
        )
        
        return ProcessFileResponse(
            message=f"PDF processed successfully. Category: {result.get('category', 'general')}, Language: {result.get('language', 'english')}",
            file_id=file_id,
            text_extracted=result['text'],
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_policies(request: QueryRequest):
    """Query the policy database with Bengali language support."""
    try:
        if not llm_interface or not policy_assistant:
            raise HTTPException(status_code=500, detail="LLM interface not available")
        
        # Search for relevant documents
        results = vector_store.search(request.question, top_k=request.top_k, threshold=request.similarity_threshold)
        
        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant information in the policy documents to answer your question. Please try rephrasing your question or contact HR for assistance.",
                sources=[],
                confidence=0.0,
                language="english"
            )
        
        # Extract context from results
        context_documents = [result['text'] for result in results]
        
        # Generate answer using policy assistant
        answer = policy_assistant.answer_policy_question(request.question, context_documents)
        
        # Detect language of the question
        language = llm_interface.detect_language(request.question)
        
        # Calculate confidence based on similarity scores
        confidence = sum(result['similarity'] for result in results) / len(results) if results else 0.0
        
        return QueryResponse(
            answer=answer,
            sources=[{
                "text": result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                "metadata": result['metadata'],
                "similarity": result['similarity']
            } for result in results],
            confidence=confidence,
            language=language
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get all documents endpoint
@app.get("/documents")
async def get_documents():
    """Get all documents in the vector store."""
    try:
        documents = vector_store.get_all_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete document endpoint
@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """Delete a document from the vector store."""
    try:
        success = vector_store.delete_document(file_id)
        if success:
            return {"message": f"Document {file_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {file_id} not found")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Clear all documents endpoint
@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store."""
    try:
        vector_store.clear_all()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get model information endpoint
@app.get("/model-info")
async def get_model_info():
    """Get information about the current LLM model."""
    try:
        if llm_interface:
            return llm_interface.get_model_info()
        else:
            raise HTTPException(status_code=500, detail="LLM interface not available")
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Test Bengali support endpoint
@app.post("/test-bengali")
async def test_bengali_support(text: str = Form(...)):
    """Test Bengali language detection and processing."""
    try:
        if not llm_interface:
            raise HTTPException(status_code=500, detail="LLM interface not available")
        
        # Detect language
        language = llm_interface.detect_language(text)
        
        # Generate a simple response
        response = llm_interface.generate_response(f"Test response for: {text}")
        
        return {
            "input_text": text,
            "detected_language": language,
            "response": response,
            "bengali_support": "enabled"
        }
        
    except Exception as e:
        logger.error(f"Error testing Bengali support: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SopAssist AI API",
        "version": "2.0.0",
        "features": [
            "OCR with Bengali support",
            "PDF processing with Bengali support",
            "Vector search",
            "Free LLM integration (Groq/Hugging Face)",
            "Bilingual responses (English/Bengali)",
            "Policy categorization"
        ],
        "endpoints": {
            "health": "/health",
            "process_image": "/process-image",
            "process_pdf": "/process-pdf",
            "query": "/query",
            "documents": "/documents",
            "model_info": "/model-info",
            "test_bengali": "/test-bengali"
        }
    }

if __name__ == "__main__":
    logger.info("Starting SopAssist AI API server...")
    uvicorn.run(
        "main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
        workers=API_CONFIG["workers"]
    ) 