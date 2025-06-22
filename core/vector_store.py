"""
Vector Store Manager for policy document embeddings and search
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
import sys
import json

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CHROMA_SETTINGS, EMBEDDING_MODEL, EMBEDDING_DIMENSION, SEARCH_CONFIG


class VectorStore:
    """Manages document embeddings and similarity search."""
    
    def __init__(self, collection_name: str = "policy_documents"):
        """Initialize the vector store."""
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=CHROMA_SETTINGS["persist_directory"],
            settings=Settings(
                anonymized_telemetry=CHROMA_SETTINGS["anonymized_telemetry"]
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Policy document embeddings"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def create_embeddings(self, texts: List[str], metadata: List[Dict] = None) -> List[List[float]]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            metadata: Optional metadata for each text
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Convert to list format for ChromaDB
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Created embeddings for {len(texts)} texts")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def add_documents(self, texts: List[str], ids: List[str], metadata: List[Dict] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text strings
            ids: List of unique IDs for each text
            metadata: Optional metadata for each text
        """
        try:
            # Create embeddings
            embeddings = self.create_embeddings(texts, metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
                metadatas=metadata or [{}] * len(texts)
            )
            
            logger.info(f"Added {len(texts)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self, query: str, top_k: int = None, threshold: float = None) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of search results with documents and scores
        """
        try:
            # Use default values if not provided
            top_k = top_k or SEARCH_CONFIG["top_k"]
            threshold = threshold or SEARCH_CONFIG["similarity_threshold"]
            
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= threshold:
                    search_results.append({
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": similarity,
                        "distance": distance
                    })
            
            logger.info(f"Found {len(search_results)} relevant documents for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_dimension": EMBEDDING_DIMENSION
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def update_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Update a single document."""
        try:
            embedding = self.create_embeddings([text])[0]
            
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}]
            )
            
            logger.info(f"Updated document: {doc_id}")
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise


class DocumentProcessor:
    """Processes documents for vector storage."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize with a vector store."""
        self.vector_store = vector_store
    
    def process_text_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_document(self, text: str, doc_id: str, metadata: Dict = None):
        """
        Add a document to the vector store with chunking.
        
        Args:
            text: Document text
            doc_id: Document ID
            metadata: Document metadata
        """
        # Split into chunks
        chunks = self.process_text_chunks(text)
        
        # Create IDs for chunks
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Create metadata for chunks
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "original_doc_id": doc_id
            })
            chunk_metadata.append(chunk_meta)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, chunk_ids, chunk_metadata)
        
        logger.info(f"Added document {doc_id} as {len(chunks)} chunks")
    
    def search_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search documents and group by original document.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of grouped search results
        """
        results = self.vector_store.search(query, top_k)
        
        # Group by original document
        grouped_results = {}
        for result in results:
            original_doc_id = result["metadata"].get("original_doc_id", "unknown")
            
            if original_doc_id not in grouped_results:
                grouped_results[original_doc_id] = {
                    "doc_id": original_doc_id,
                    "chunks": [],
                    "max_similarity": 0,
                    "total_chunks": 0
                }
            
            grouped_results[original_doc_id]["chunks"].append(result)
            grouped_results[original_doc_id]["max_similarity"] = max(
                grouped_results[original_doc_id]["max_similarity"],
                result["similarity"]
            )
            grouped_results[original_doc_id]["total_chunks"] = result["metadata"].get("total_chunks", 1)
        
        # Sort by max similarity
        sorted_results = sorted(
            grouped_results.values(),
            key=lambda x: x["max_similarity"],
            reverse=True
        )
        
        return sorted_results


def main():
    """Test the vector store functionality."""
    # Initialize vector store
    vector_store = VectorStore()
    
    # Test documents
    test_docs = [
        "Employees are entitled to 20 vacation days per year.",
        "All employees must complete security training annually.",
        "Remote work is allowed up to 3 days per week.",
        "Expense reports must be submitted within 30 days."
    ]
    
    test_ids = [f"doc_{i}" for i in range(len(test_docs))]
    test_metadata = [{"category": "hr", "type": "policy"} for _ in test_docs]
    
    # Add documents
    vector_store.add_documents(test_docs, test_ids, test_metadata)
    
    # Test search
    query = "How many vacation days do I get?"
    results = vector_store.search(query)
    
    print(f"Search results for: {query}")
    for result in results:
        print(f"Document: {result['document']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Metadata: {result['metadata']}")
        print()


if __name__ == "__main__":
    main() 