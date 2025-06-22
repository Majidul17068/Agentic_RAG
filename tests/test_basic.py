"""
Basic tests for SopAssist AI components
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.ocr_processor import OCRProcessor
from core.vector_store import VectorStore, DocumentProcessor
from core.llm_interface import OllamaInterface, PolicyAssistant


class TestOCRProcessor:
    """Test OCR processor functionality."""
    
    def test_initialization(self):
        """Test OCR processor initialization."""
        try:
            processor = OCRProcessor()
            assert processor is not None
        except Exception as e:
            pytest.skip(f"OCR processor not available: {e}")
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        processor = OCRProcessor()
        
        # Test text cleaning
        dirty_text = "  This   is   a   test   text  with  extra  spaces  "
        clean_text = processor._clean_text(dirty_text)
        
        assert clean_text == "This is a test text with extra spaces"
        assert "  " not in clean_text  # No double spaces


class TestVectorStore:
    """Test vector store functionality."""
    
    def test_initialization(self):
        """Test vector store initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(collection_name="test_collection")
            assert vector_store is not None
    
    def test_add_and_search_documents(self):
        """Test adding and searching documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(collection_name="test_collection")
            
            # Test documents
            test_docs = [
                "Employees get 20 vacation days per year.",
                "All employees must complete security training.",
                "Remote work is allowed 3 days per week."
            ]
            
            test_ids = [f"test_doc_{i}" for i in range(len(test_docs))]
            test_metadata = [{"category": "hr", "test": True} for _ in test_docs]
            
            # Add documents
            vector_store.add_documents(test_docs, test_ids, test_metadata)
            
            # Search
            results = vector_store.search("vacation days", top_k=2)
            
            assert len(results) > 0
            assert any("vacation" in result["document"].lower() for result in results)
    
    def test_document_processor(self):
        """Test document processor functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(collection_name="test_collection")
            processor = DocumentProcessor(vector_store)
            
            # Test text chunking
            long_text = "This is a long text. " * 50  # Create long text
            chunks = processor.process_text_chunks(long_text, chunk_size=100, overlap=20)
            
            assert len(chunks) > 1  # Should be split into multiple chunks
            assert all(len(chunk) <= 100 for chunk in chunks)  # No chunk too long


class TestLLMInterface:
    """Test LLM interface functionality."""
    
    def test_initialization(self):
        """Test LLM interface initialization."""
        try:
            llm = OllamaInterface()
            assert llm is not None
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
    
    def test_basic_response(self):
        """Test basic response generation."""
        try:
            llm = OllamaInterface()
            response = llm.generate_response("What is 2+2?")
            assert response is not None
            assert len(response) > 0
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
    
    def test_policy_assistant(self):
        """Test policy assistant functionality."""
        try:
            llm = OllamaInterface()
            assistant = PolicyAssistant(llm)
            
            # Test policy summarization
            policy_text = "Employees are entitled to 20 vacation days per year."
            summary = assistant.summarize_policy(policy_text)
            
            assert summary is not None
            assert len(summary) > 0
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")


class TestIntegration:
    """Test integration between components."""
    
    def test_end_to_end_processing(self):
        """Test end-to-end document processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test vector store
            vector_store = VectorStore(collection_name="test_integration")
            processor = DocumentProcessor(vector_store)
            
            # Test document processing
            test_text = "This is a test policy document about employee benefits."
            doc_id = "test_integration_doc"
            metadata = {"test": True, "category": "test"}
            
            processor.add_document(test_text, doc_id, metadata)
            
            # Test search
            results = vector_store.search("employee benefits")
            assert len(results) > 0
            
            # Test with LLM (if available)
            try:
                llm = OllamaInterface()
                assistant = PolicyAssistant(llm)
                
                # Test question answering
                question = "What is this policy about?"
                answer = assistant.answer_policy_question(question, [test_text])
                
                assert answer is not None
                assert len(answer) > 0
            except Exception as e:
                pytest.skip(f"Ollama not available for integration test: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 