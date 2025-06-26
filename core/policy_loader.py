"""
Policy Loader for loading extracted policies from text file
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

class PolicyLoader:
    """Loads and manages extracted policies from text file."""
    
    def __init__(self, policy_file_path: str = "data/all_policies.txt"):
        """Initialize the policy loader."""
        self.policy_file_path = Path(policy_file_path)
        self.policies = {}
        self.load_policies()
    
    def load_policies(self):
        """Load all policies from the text file."""
        if not self.policy_file_path.exists():
            logger.warning(f"Policy file not found: {self.policy_file_path}")
            return
        
        try:
            with open(self.policy_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content by policy separators
            policy_sections = re.split(r'={80}', content)
            
            for section in policy_sections:
                if not section.strip():
                    continue
                
                # Extract policy name and content
                lines = section.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # Find policy name from header
                policy_name = None
                filename = None
                content_start = 0
                
                for i, line in enumerate(lines):
                    if line.startswith('POLICY:'):
                        policy_name = line.replace('POLICY:', '').strip()
                    elif line.startswith('FILENAME:'):
                        filename = line.replace('FILENAME:', '').strip()
                    elif line.strip() and not line.startswith('POLICY:') and not line.startswith('FILENAME:'):
                        content_start = i
                        break
                
                if policy_name and filename:
                    # Extract the actual policy content
                    policy_content = '\n'.join(lines[content_start:]).strip()
                    
                    if policy_content:
                        self.policies[filename] = {
                            'name': policy_name,
                            'filename': filename,
                            'content': policy_content,
                            'length': len(policy_content)
                        }
            
            logger.info(f"Loaded {len(self.policies)} policies from {self.policy_file_path}")
            
        except Exception as e:
            logger.error(f"Error loading policies: {e}")
    
    def get_policy_by_filename(self, filename: str) -> Optional[Dict]:
        """Get a specific policy by filename."""
        return self.policies.get(filename)
    
    def get_all_policies(self) -> Dict:
        """Get all loaded policies."""
        return self.policies
    
    def search_policies(self, query: str) -> List[Dict]:
        """Search policies by content."""
        results = []
        query_lower = query.lower()
        
        for filename, policy in self.policies.items():
            if query_lower in policy['content'].lower():
                results.append({
                    'filename': filename,
                    'name': policy['name'],
                    'content': policy['content'],
                    'relevance': 'exact_match'
                })
        
        return results
    
    def get_policy_summary(self) -> Dict:
        """Get summary of loaded policies."""
        total_policies = len(self.policies)
        total_content_length = sum(p['length'] for p in self.policies.values())
        
        return {
            'total_policies': total_policies,
            'total_content_length': total_content_length,
            'average_length': total_content_length / total_policies if total_policies > 0 else 0,
            'policy_names': list(self.policies.keys())
        }
    
    def get_policy_content_for_llm(self, query: str = None) -> str:
        """Get formatted policy content for LLM context."""
        if not self.policies:
            return "No policies loaded."
        
        if query:
            # If query provided, try to find relevant policies
            relevant_policies = self.search_policies(query)
            if relevant_policies:
                content_parts = []
                for policy in relevant_policies:
                    content_parts.append(f"POLICY: {policy['name']}\n{policy['content']}\n")
                return "\n".join(content_parts)
        
        # Return all policies if no query or no relevant policies found
        content_parts = []
        for filename, policy in self.policies.items():
            content_parts.append(f"POLICY: {policy['name']}\n{policy['content']}\n")
        
        return "\n".join(content_parts) 