#!/usr/bin/env python3
"""
Script to add sample policies to the vector store for testing
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.vector_store import VectorStore, DocumentProcessor


def add_sample_policies():
    """Add sample policies to the vector store."""
    try:
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore()
        document_processor = DocumentProcessor(vector_store)
        
        # Sample policies
        policies = {
            "vacation_policy": """VACATION POLICY

Employees are entitled to 20 vacation days per year. Vacation requests must be submitted at least 2 weeks in advance. Unused vacation days may be carried over to the next year up to a maximum of 5 days. Vacation days are prorated for new employees based on their start date.""",
            
            "expense_policy": """EXPENSE REPORT POLICY

All expense reports must be submitted within 30 days of the expense being incurred. Receipts must be attached for all expenses over $25. Approved expenses will be reimbursed within 2 weeks of submission. Unapproved expenses will be returned with explanation.""",
            
            "remote_work_policy": """REMOTE WORK POLICY

Employees may work remotely up to 3 days per week with manager approval. Remote work days must be scheduled in advance. Employees must be available during normal business hours and maintain the same level of productivity. All company equipment must be properly secured when working remotely.""",
            
            "security_policy": """SECURITY TRAINING POLICY

All employees must complete annual security training. Training must be completed within 30 days of the annual due date. New employees must complete security training within their first 30 days. Failure to complete training may result in restricted system access.""",
            
            "sick_leave_policy": """SICK LEAVE POLICY

Employees are entitled to 10 sick days per year. Sick leave may be used for personal illness or medical appointments. Sick leave requests should be submitted as soon as possible. Extended sick leave may require medical documentation.""",
            
            "workplace_conduct_policy": """WORKPLACE CONDUCT POLICY

All employees must maintain professional behavior in the workplace. Harassment, discrimination, or inappropriate conduct will not be tolerated. Employees should report any concerns to HR immediately. Violations may result in disciplinary action up to and including termination.""",
            
            "data_protection_policy": """DATA PROTECTION POLICY

All company data must be handled according to security guidelines. Personal information must be kept confidential. Data should only be accessed on a need-to-know basis. Unauthorized access or sharing of data may result in disciplinary action.""",
            
            "meeting_policy": """MEETING POLICY

Meetings should have a clear agenda and purpose. All meetings should start and end on time. Meeting participants should come prepared and contribute constructively. Follow-up actions should be documented and assigned to specific individuals.""",
            
            "equipment_policy": """EQUIPMENT POLICY

Company equipment is provided for business use only. Equipment must be properly maintained and secured. Personal use of company equipment is not permitted. Equipment must be returned upon termination of employment.""",
            
            "travel_policy": """TRAVEL POLICY

Business travel must be pre-approved by management. Travel expenses should be reasonable and necessary. Employees should use the most cost-effective travel options available. All travel expenses must be documented and submitted for reimbursement."""
        }
        
        # Add each policy to the vector store
        for policy_name, policy_text in policies.items():
            try:
                metadata = {
                    "category": "sample_policy",
                    "policy_type": policy_name,
                    "source": "sample_data",
                    "text_length": len(policy_text)
                }
                
                document_processor.add_document(policy_text, policy_name, metadata)
                logger.info(f"Added policy: {policy_name}")
                
            except Exception as e:
                logger.error(f"Error adding policy {policy_name}: {e}")
        
        # Get final stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Vector store now contains {stats.get('total_documents', 0)} documents")
        
        logger.info("✅ Sample policies added successfully!")
        
        # Test search
        logger.info("Testing search functionality...")
        test_query = "vacation days"
        results = vector_store.search(test_query, top_k=3)
        
        if results:
            logger.info(f"Search test successful - found {len(results)} results for '{test_query}'")
            for i, result in enumerate(results, 1):
                logger.info(f"Result {i}: {result['document'][:100]}... (similarity: {result['similarity']:.3f})")
        else:
            logger.warning("Search test returned no results")
        
    except Exception as e:
        logger.error(f"Error adding sample policies: {e}")
        raise


def main():
    """Main function."""
    logger.info("Adding sample policies to vector store...")
    add_sample_policies()
    logger.info("Done!")


if __name__ == "__main__":
    main() 