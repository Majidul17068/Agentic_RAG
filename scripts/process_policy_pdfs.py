#!/usr/bin/env python3
"""
Script to process PDF policy files and add them to the vector store
"""

import sys
from pathlib import Path
import argparse
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import POLICY_DIR, PROCESSED_PDFS_DIR
from core.pdf_processor import PDFProcessor
from core.vector_store import VectorStore, DocumentProcessor


def process_policy_pdfs(input_dir: Path = None, output_dir: Path = None, 
                       collection_name: str = "policy_documents"):
    """
    Process all PDF policy files and add them to the vector store.
    
    Args:
        input_dir: Directory containing PDF files (default: Policy file/)
        output_dir: Directory to save processed text (default: data/processed_pdfs)
        collection_name: Name of the vector store collection
    """
    # Use defaults if not provided
    input_dir = input_dir or POLICY_DIR
    output_dir = output_dir or PROCESSED_PDFS_DIR
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing PDF policies from: {input_dir}")
    logger.info(f"Saving processed text to: {output_dir}")
    
    try:
        # Initialize components
        logger.info("Initializing PDF processor...")
        pdf_processor = PDFProcessor()
        
        logger.info("Initializing vector store...")
        vector_store = VectorStore(collection_name)
        
        logger.info("Initializing document processor...")
        document_processor = DocumentProcessor(vector_store)
        
        # Process PDF files
        logger.info("Processing PDF files...")
        results = pdf_processor.process_policy_directory(input_dir, output_dir)
        
        if not results:
            logger.warning("No PDF files found to process")
            return
        
        logger.info(f"Successfully extracted text from {len(results)} PDF files")
        
        # Add to vector store
        logger.info("Adding documents to vector store...")
        for filename, result in results.items():
            try:
                # Create document ID
                doc_id = f"pdf_{filename.replace('.pdf', '')}"
                
                # Get categorization
                categorization = result["categorization"]
                
                # Create metadata
                metadata = {
                    "source": "pdf_policy",
                    "original_file": filename,
                    "text_length": result["text_length"],
                    "file_size": result["file_size"],
                    "processing_method": "pdf_extraction",
                    "categories": categorization["categories"],
                    "year": categorization["year"],
                    "policy_type": categorization["policy_type"]
                }
                
                # Add to vector store
                document_processor.add_document(result["text"], doc_id, metadata)
                logger.info(f"Added document: {doc_id} -> {categorization['categories']}")
                
            except Exception as e:
                logger.error(f"Error adding document {filename}: {e}")
        
        # Get final stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store now contains {stats.get('total_documents', 0)} documents")
        
        # Generate summary
        summary = pdf_processor.get_policy_summary(results)
        logger.info("📊 Processing Summary:")
        logger.info(f"Total files: {summary['total_files']}")
        logger.info(f"Total text length: {summary['total_text_length']:,} characters")
        logger.info(f"Average text length: {summary['average_text_length']:,.0f} characters")
        
        logger.info("📂 Category Distribution:")
        for category, count in summary['category_distribution'].items():
            logger.info(f"  {category}: {count} files")
        
        logger.info("📅 Year Distribution:")
        for year, count in summary['year_distribution'].items():
            logger.info(f"  {year}: {count} files")
        
        logger.info("✅ PDF policy processing complete!")
        
    except Exception as e:
        logger.error(f"Error processing PDF policies: {e}")
        raise


def process_single_pdf(pdf_path: Path, collection_name: str = "policy_documents"):
    """
    Process a single PDF policy file.
    
    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the vector store collection
    """
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    logger.info(f"Processing single PDF: {pdf_path}")
    
    try:
        # Initialize components
        pdf_processor = PDFProcessor()
        vector_store = VectorStore(collection_name)
        document_processor = DocumentProcessor(vector_store)
        
        # Extract text
        text = pdf_processor.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return
        
        # Categorize policy
        categorization = pdf_processor.categorize_policy(pdf_path.name)
        
        # Create document ID and metadata
        doc_id = f"single_pdf_{pdf_path.stem}"
        metadata = {
            "source": "single_pdf",
            "original_file": pdf_path.name,
            "text_length": len(text),
            "file_size": pdf_path.stat().st_size,
            "processing_method": "pdf_extraction",
            "categories": categorization["categories"],
            "year": categorization["year"],
            "policy_type": categorization["policy_type"]
        }
        
        # Add to vector store
        document_processor.add_document(text, doc_id, metadata)
        logger.info(f"Successfully added document: {doc_id}")
        
        # Save extracted text
        output_file = PROCESSED_PDFS_DIR / f"{pdf_path.stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved extracted text to: {output_file}")
        
        # Show categorization
        logger.info(f"Policy categorized as: {categorization['categories']}")
        logger.info(f"Policy type: {categorization['policy_type']}")
        logger.info(f"Year: {categorization['year']}")
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        raise


def list_policy_documents(collection_name: str = "policy_documents"):
    """List all policy documents in the vector store."""
    try:
        vector_store = VectorStore(collection_name)
        stats = vector_store.get_collection_stats()
        
        logger.info(f"Collection: {stats.get('collection_name', 'N/A')}")
        logger.info(f"Total documents: {stats.get('total_documents', 0)}")
        logger.info(f"Embedding dimension: {stats.get('embedding_dimension', 'N/A')}")
        
        # Search for some sample documents to show categories
        sample_results = vector_store.search("policy", top_k=10)
        
        if sample_results:
            logger.info("📄 Sample Policy Documents:")
            for i, result in enumerate(sample_results, 1):
                metadata = result.get('metadata', {})
                categories = metadata.get('categories', ['unknown'])
                policy_type = metadata.get('policy_type', 'unknown')
                year = metadata.get('year', 'unknown')
                
                logger.info(f"  {i}. {policy_type} ({year}) - Categories: {', '.join(categories)}")
        
    except Exception as e:
        logger.error(f"Error listing policy documents: {e}")


def test_search_functionality(collection_name: str = "policy_documents"):
    """Test search functionality with sample queries."""
    try:
        vector_store = VectorStore(collection_name)
        
        test_queries = [
            "vacation policy",
            "travel allowance",
            "medical benefits",
            "salary increment",
            "house rent"
        ]
        
        logger.info("🔍 Testing Search Functionality:")
        
        for query in test_queries:
            results = vector_store.search(query, top_k=3)
            logger.info(f"\nQuery: '{query}'")
            logger.info(f"Found {len(results)} results")
            
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                policy_type = metadata.get('policy_type', 'unknown')
                similarity = result.get('similarity', 0)
                logger.info(f"  {i}. {policy_type} (similarity: {similarity:.3f})")
        
    except Exception as e:
        logger.error(f"Error testing search functionality: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Process PDF policy files and add to vector store")
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        default=POLICY_DIR,
        help="Directory containing PDF policy files"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=PROCESSED_PDFS_DIR,
        help="Directory to save processed text"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="policy_documents",
        help="Vector store collection name"
    )
    parser.add_argument(
        "--single-pdf", 
        type=Path,
        help="Process a single PDF file"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all policy documents in the vector store"
    )
    parser.add_argument(
        "--test-search", 
        action="store_true",
        help="Test search functionality with sample queries"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    try:
        if args.list:
            list_policy_documents(args.collection)
        elif args.test_search:
            test_search_functionality(args.collection)
        elif args.single_pdf:
            process_single_pdf(args.single_pdf, args.collection)
        else:
            process_policy_pdfs(args.input_dir, args.output_dir, args.collection)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 