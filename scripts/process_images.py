#!/usr/bin/env python3
"""
Script to process policy images and add them to the vector store
"""

import sys
from pathlib import Path
import argparse
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import IMAGES_DIR, PROCESSED_DIR
from core.ocr_processor import OCRProcessor
from core.vector_store import VectorStore, DocumentProcessor


def process_images(input_dir: Path = None, output_dir: Path = None, 
                  collection_name: str = "policy_documents"):
    """
    Process all images in the input directory and add them to the vector store.
    
    Args:
        input_dir: Directory containing images (default: data/images)
        output_dir: Directory to save processed text (default: data/processed)
        collection_name: Name of the vector store collection
    """
    # Use defaults if not provided
    input_dir = input_dir or IMAGES_DIR
    output_dir = output_dir or PROCESSED_DIR
    
    # Ensure directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing images from: {input_dir}")
    logger.info(f"Saving processed text to: {output_dir}")
    
    try:
        # Initialize components
        logger.info("Initializing OCR processor...")
        ocr_processor = OCRProcessor()
        
        logger.info("Initializing vector store...")
        vector_store = VectorStore(collection_name)
        
        logger.info("Initializing document processor...")
        document_processor = DocumentProcessor(vector_store)
        
        # Process images
        logger.info("Processing images...")
        results = ocr_processor.process_directory(input_dir, output_dir)
        
        if not results:
            logger.warning("No images found to process")
            return
        
        logger.info(f"Successfully extracted text from {len(results)} images")
        
        # Add to vector store
        logger.info("Adding documents to vector store...")
        for image_name, text in results.items():
            try:
                # Create document ID
                doc_id = f"processed_{image_name}"
                
                # Create metadata
                metadata = {
                    "source": "processed_image",
                    "original_image": image_name,
                    "text_length": len(text),
                    "processing_method": "ocr"
                }
                
                # Add to vector store
                document_processor.add_document(text, doc_id, metadata)
                logger.info(f"Added document: {doc_id}")
                
            except Exception as e:
                logger.error(f"Error adding document {image_name}: {e}")
        
        # Get final stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store now contains {stats.get('total_documents', 0)} documents")
        
        logger.info("✅ Image processing complete!")
        
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise


def process_single_image(image_path: Path, collection_name: str = "policy_documents"):
    """
    Process a single image and add it to the vector store.
    
    Args:
        image_path: Path to the image file
        collection_name: Name of the vector store collection
    """
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return
    
    logger.info(f"Processing single image: {image_path}")
    
    try:
        # Initialize components
        ocr_processor = OCRProcessor()
        vector_store = VectorStore(collection_name)
        document_processor = DocumentProcessor(vector_store)
        
        # Extract text
        text = ocr_processor.extract_text(image_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from {image_path}")
            return
        
        # Create document ID and metadata
        doc_id = f"single_{image_path.stem}"
        metadata = {
            "source": "single_image",
            "original_image": image_path.name,
            "text_length": len(text),
            "processing_method": "ocr"
        }
        
        # Add to vector store
        document_processor.add_document(text, doc_id, metadata)
        logger.info(f"Successfully added document: {doc_id}")
        
        # Save extracted text
        output_file = PROCESSED_DIR / f"{image_path.stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved extracted text to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        raise


def list_processed_documents(collection_name: str = "policy_documents"):
    """List all documents in the vector store."""
    try:
        vector_store = VectorStore(collection_name)
        stats = vector_store.get_collection_stats()
        
        logger.info(f"Collection: {stats.get('collection_name', 'N/A')}")
        logger.info(f"Total documents: {stats.get('total_documents', 0)}")
        logger.info(f"Embedding dimension: {stats.get('embedding_dimension', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Process policy images and add to vector store")
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        default=IMAGES_DIR,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=PROCESSED_DIR,
        help="Directory to save processed text"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="policy_documents",
        help="Vector store collection name"
    )
    parser.add_argument(
        "--single-image", 
        type=Path,
        help="Process a single image file"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all documents in the vector store"
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
            list_processed_documents(args.collection)
        elif args.single_image:
            process_single_image(args.single_image, args.collection)
        else:
            process_images(args.input_dir, args.output_dir, args.collection)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 