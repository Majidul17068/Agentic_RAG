"""
Enhanced Policy Agents for Agentic RAG with PDF Policy Documents
"""

from crewai import Agent, Task, Crew, Process
from typing import List, Dict, Optional, Any
from loguru import logger
import sys
from pathlib import Path
import re

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CREWAI_CONFIG, POLICY_CATEGORIES
from core.vector_store import VectorStore
from core.free_llm_interface import FreeLLMInterface, PolicyAssistant


class EnhancedPolicyAgents:
    """Enhanced CrewAI agents for policy analysis with PDF documents and free LLMs."""
    
    def __init__(self, vector_store: VectorStore, llm_provider: str = "groq", model: str = None):
        """
        Initialize the enhanced policy agents.
        
        Args:
            vector_store: Vector store instance
            llm_provider: "groq", "huggingface", or "ollama"
            model: Specific model name
        """
        self.vector_store = vector_store
        self.config = CREWAI_CONFIG
        
        # Initialize LLM interface
        if llm_provider in ["groq", "huggingface"]:
            self.llm_interface = FreeLLMInterface(llm_provider, model)
        else:
            # Fallback to Ollama
            from core.llm_interface import OllamaInterface
            self.llm_interface = OllamaInterface(model)
        
        # Initialize policy assistant
        self.policy_assistant = PolicyAssistant(self.llm_interface)
        
        # Initialize agents
        self._create_agents()
    
    def _create_agents(self):
        """Create the specialized agents for policy analysis."""
        
        # Policy Research Agent - Finds relevant policy documents
        self.research_agent = Agent(
            role="Policy Research Specialist",
            goal="Find the most relevant policy documents and information to answer user questions",
            backstory="""You are an expert at analyzing policy documents and finding relevant information. 
            You have years of experience in HR, legal, and corporate policy analysis. You can quickly 
            identify which policies are relevant to specific questions and extract the most important 
            information from them. You understand the categorization of policies and can find related 
            documents across different policy types.""",
            verbose=self.config["verbose"],
            allow_delegation=False,
            tools=[self._search_policies_tool, self._categorize_question_tool],
            llm=self.llm_interface
        )
        
        # Policy Analysis Agent - Analyzes and interprets policy information
        self.analysis_agent = Agent(
            role="Policy Analysis Expert",
            goal="Analyze policy information and provide clear, accurate interpretations",
            backstory="""You are a senior policy analyst with expertise in interpreting complex policy 
            documents. You can break down complicated policies into clear, understandable explanations 
            and identify the key points that are most relevant to specific situations. You have a 
            strong background in legal interpretation and corporate compliance. You can extract 
            specific requirements, deadlines, procedures, and exceptions from policy documents.""",
            verbose=self.config["verbose"],
            allow_delegation=False,
            tools=[self._analyze_policy_tool, self._extract_requirements_tool],
            llm=self.llm_interface
        )
        
        # Policy Synthesis Agent - Combines information from multiple sources
        self.synthesis_agent = Agent(
            role="Policy Synthesis Specialist",
            goal="Combine information from multiple policy sources into coherent answers",
            backstory="""You are an expert at synthesizing information from multiple policy documents 
            into clear, comprehensive answers. You can identify conflicts, overlaps, and gaps between 
            different policies. You understand how different policies relate to each other and can 
            provide holistic answers that consider all relevant policy aspects.""",
            verbose=self.config["verbose"],
            allow_delegation=False,
            tools=[self._synthesize_policies_tool, self._identify_conflicts_tool],
            llm=self.llm_interface
        )
        
        # Policy Communication Agent - Provides clear, user-friendly responses
        self.communication_agent = Agent(
            role="Policy Communication Specialist",
            goal="Provide clear, helpful, and professional responses to policy questions",
            backstory="""You are an expert communicator who specializes in explaining complex policy 
            information in a clear, user-friendly way. You have excellent writing skills and can 
            present information in a professional yet approachable manner. You always ensure that 
            responses are accurate, complete, and easy to understand. You can format responses 
            appropriately for different audiences and include proper citations.""",
            verbose=self.config["verbose"],
            allow_delegation=False,
            tools=[self._format_response_tool, self._add_citations_tool],
            llm=self.llm_interface
        )
    
    def _search_policies_tool(self, query: str, category: str = None) -> str:
        """Tool for searching policy documents with optional category filtering."""
        try:
            # Search with category filter if provided
            if category and category in POLICY_CATEGORIES:
                # Add category keywords to query
                category_keywords = " ".join(POLICY_CATEGORIES[category])
                enhanced_query = f"{query} {category_keywords}"
                results = self.vector_store.search(enhanced_query, top_k=5)
            else:
                results = self.vector_store.search(query, top_k=5)
            
            if not results:
                return "No relevant policy documents found for this query."
            
            # Format results with metadata
            formatted_results = []
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                categories = metadata.get('categories', ['unknown'])
                policy_type = metadata.get('policy_type', 'unknown')
                year = metadata.get('year', 'unknown')
                
                formatted_results.append(
                    f"Document {i}:\n"
                    f"Content: {result['document']}\n"
                    f"Categories: {', '.join(categories)}\n"
                    f"Policy Type: {policy_type}\n"
                    f"Year: {year}\n"
                    f"Relevance Score: {result['similarity']:.3f}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error in search_policies_tool: {e}")
            return f"Error searching policies: {str(e)}"
    
    def _categorize_question_tool(self, question: str) -> str:
        """Tool for categorizing user questions to find relevant policy types."""
        try:
            question_lower = question.lower()
            
            # Find matching categories
            matched_categories = []
            for category, keywords in POLICY_CATEGORIES.items():
                if any(keyword in question_lower for keyword in keywords):
                    matched_categories.append(category)
            
            if matched_categories:
                return f"Question matches these policy categories: {', '.join(matched_categories)}"
            else:
                return "Question doesn't clearly match any specific policy category. Searching general policies."
                
        except Exception as e:
            logger.error(f"Error in categorize_question_tool: {e}")
            return f"Error categorizing question: {str(e)}"
    
    def _analyze_policy_tool(self, policy_text: str) -> str:
        """Tool for analyzing policy text and extracting key information."""
        try:
            # Use the policy assistant to analyze the text
            summary = self.policy_assistant.summarize_policy(policy_text)
            key_points = self.policy_assistant.extract_key_points(policy_text)
            
            # Create analysis
            analysis = f"""Policy Analysis:

Summary:
{summary}

Key Points:
{chr(10).join(f"- {point}" for point in key_points)}

This analysis provides the essential information from the policy document in a clear, structured format."""
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_policy_tool: {e}")
            return f"Error analyzing policy: {str(e)}"
    
    def _extract_requirements_tool(self, policy_text: str) -> str:
        """Tool for extracting specific requirements and procedures from policy text."""
        try:
            prompt = f"""Extract specific requirements, deadlines, procedures, and exceptions from this policy:

{policy_text}

Please identify:
1. Specific requirements or rules
2. Deadlines or timeframes
3. Procedures or steps to follow
4. Exceptions or special conditions
5. Consequences or penalties

Format as a structured list:"""
            
            response = self.llm_interface.generate_response(prompt, max_tokens=1000)
            return response
            
        except Exception as e:
            logger.error(f"Error in extract_requirements_tool: {e}")
            return f"Error extracting requirements: {str(e)}"
    
    def _synthesize_policies_tool(self, policy_texts: List[str]) -> str:
        """Tool for synthesizing information from multiple policy documents."""
        try:
            combined_text = "\n\n---\n\n".join(policy_texts)
            
            prompt = f"""Synthesize information from these multiple policy documents:

{combined_text}

Please provide:
1. A comprehensive overview
2. How these policies relate to each other
3. Any conflicts or overlaps
4. Key takeaways for users

Focus on creating a coherent understanding:"""
            
            response = self.llm_interface.generate_response(prompt, max_tokens=1500)
            return response
            
        except Exception as e:
            logger.error(f"Error in synthesize_policies_tool: {e}")
            return f"Error synthesizing policies: {str(e)}"
    
    def _identify_conflicts_tool(self, policy_texts: List[str]) -> str:
        """Tool for identifying conflicts between different policies."""
        try:
            combined_text = "\n\n---\n\n".join(policy_texts)
            
            prompt = f"""Analyze these policy documents for potential conflicts or inconsistencies:

{combined_text}

Please identify:
1. Any conflicting requirements
2. Inconsistent procedures
3. Overlapping but different rules
4. Gaps in coverage
5. Recommendations for resolution

Be specific about where conflicts occur:"""
            
            response = self.llm_interface.generate_response(prompt, max_tokens=1000)
            return response
            
        except Exception as e:
            logger.error(f"Error in identify_conflicts_tool: {e}")
            return f"Error identifying conflicts: {str(e)}"
    
    def _format_response_tool(self, content: str) -> str:
        """Tool for formatting the final response professionally."""
        try:
            prompt = f"""Format the following policy information into a clear, professional response:

{content}

Please ensure the response is:
1. Well-structured with clear sections
2. Professional in tone
3. Accurate and complete
4. Easy to read and understand
5. Includes relevant policy references

Formatted Response:"""
            
            response = self.llm_interface.generate_response(prompt, max_tokens=1000)
            return response
            
        except Exception as e:
            logger.error(f"Error in format_response_tool: {e}")
            return content
    
    def _add_citations_tool(self, content: str, sources: List[Dict]) -> str:
        """Tool for adding proper citations to the response."""
        try:
            citations = []
            for i, source in enumerate(sources, 1):
                metadata = source.get('metadata', {})
                policy_type = metadata.get('policy_type', 'Unknown Policy')
                year = metadata.get('year', 'Unknown Year')
                citations.append(f"[{i}] {policy_type} ({year})")
            
            citations_text = "\n".join(citations)
            
            prompt = f"""Add proper citations to this response:

Content:
{content}

Sources to cite:
{citations_text}

Please integrate citations naturally into the text and add a reference section at the end:"""
            
            response = self.llm_interface.generate_response(prompt, max_tokens=1000)
            return response
            
        except Exception as e:
            logger.error(f"Error in add_citations_tool: {e}")
            return content
    
    def answer_policy_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a policy question using the enhanced agent crew.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Create tasks for the agent workflow
            research_task = Task(
                description=f"""Research and find the most relevant policy documents for this question: "{question}"
                
                Your task is to:
                1. Categorize the question to identify relevant policy types
                2. Search for policy documents that are relevant to the question
                3. Identify the most important and relevant information
                4. Provide a summary of what you found
                
                Use the categorize_question_tool first, then search_policies_tool to find relevant documents.""",
                agent=self.research_agent,
                expected_output="A comprehensive list of relevant policy documents and key information found"
            )
            
            analysis_task = Task(
                description=f"""Analyze the policy information found by the research agent for this question: "{question}"
                
                Your task is to:
                1. Review the policy documents and information provided
                2. Analyze the relevance and importance of each piece of information
                3. Extract specific requirements, procedures, and deadlines
                4. Identify key points that directly answer the question
                5. Provide a detailed analysis of the policy implications
                
                Use the analyze_policy_tool and extract_requirements_tool to help with your analysis.""",
                agent=self.analysis_agent,
                expected_output="A detailed analysis of the policy information and its relevance to the question",
                context=[research_task]
            )
            
            synthesis_task = Task(
                description=f"""Synthesize the policy analysis for this question: "{question}"
                
                Your task is to:
                1. Combine information from multiple policy sources
                2. Identify any conflicts or overlaps between policies
                3. Create a coherent understanding of the policy landscape
                4. Provide recommendations based on the analysis
                
                Use the synthesize_policies_tool and identify_conflicts_tool to help with synthesis.""",
                agent=self.synthesis_agent,
                expected_output="A synthesized understanding of the policy information with conflict analysis",
                context=[analysis_task]
            )
            
            communication_task = Task(
                description=f"""Create a clear, professional response to this question: "{question}"
                
                Your task is to:
                1. Take the synthesized analysis and create a comprehensive answer
                2. Ensure the response is clear, accurate, and helpful
                3. Format the response in a professional manner
                4. Include proper citations and references
                5. Make it easy for users to understand and act upon
                
                Use the format_response_tool and add_citations_tool to create the final response.""",
                agent=self.communication_agent,
                expected_output="A clear, professional, and comprehensive answer to the user's question",
                context=[synthesis_task]
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[self.research_agent, self.analysis_agent, self.synthesis_agent, self.communication_agent],
                tasks=[research_task, analysis_task, synthesis_task, communication_task],
                process=Process.sequential,
                verbose=self.config["verbose"],
                max_iterations=self.config["max_iterations"]
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            return {
                "answer": result,
                "question": question,
                "agents_used": ["research", "analysis", "synthesis", "communication"],
                "llm_provider": self.llm_interface.provider if hasattr(self.llm_interface, 'provider') else "ollama",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in answer_policy_question: {e}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "question": question,
                "agents_used": [],
                "status": "error",
                "error": str(e)
            }
    
    def get_policy_recommendations(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        Get policy recommendations based on a question and context.
        
        Args:
            question: The user's question
            context: Optional additional context
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            # Create research task
            research_task = Task(
                description=f"""Research policy information for this question: "{question}"
                
                Additional context: {context or "None provided"}
                
                Find relevant policy documents and information that could help answer this question.""",
                agent=self.research_agent,
                expected_output="Relevant policy documents and information"
            )
            
            # Create recommendations task
            recommendations_task = Task(
                description=f"""Based on the research, provide specific policy recommendations for this question: "{question}"
                
                Your task is to:
                1. Review the policy information found
                2. Identify specific recommendations or actions
                3. Consider best practices and compliance requirements
                4. Provide actionable advice
                5. Consider any policy conflicts or gaps
                
                Additional context: {context or "None provided"}""",
                agent=self.analysis_agent,
                expected_output="Specific policy recommendations and actionable advice",
                context=[research_task]
            )
            
            # Create formatting task
            formatting_task = Task(
                description="Format the recommendations into a clear, actionable document with proper citations.",
                agent=self.communication_agent,
                expected_output="A well-formatted document with clear recommendations and citations",
                context=[recommendations_task]
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[self.research_agent, self.analysis_agent, self.communication_agent],
                tasks=[research_task, recommendations_task, formatting_task],
                process=Process.sequential,
                verbose=self.config["verbose"]
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            return {
                "recommendations": result,
                "question": question,
                "context": context,
                "agents_used": ["research", "analysis", "communication"],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in get_policy_recommendations: {e}")
            return {
                "recommendations": f"Error generating recommendations: {str(e)}",
                "question": question,
                "context": context,
                "agents_used": [],
                "status": "error",
                "error": str(e)
            }


def main():
    """Test the enhanced policy agents."""
    try:
        # Initialize components
        vector_store = VectorStore()
        agents = EnhancedPolicyAgents(vector_store, llm_provider="groq")
        
        # Test question answering
        question = "What is the vacation policy for employees?"
        result = agents.answer_policy_question(question)
        
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Status: {result['status']}")
        print(f"LLM Provider: {result['llm_provider']}")
        
    except Exception as e:
        print(f"Error testing enhanced policy agents: {e}")


if __name__ == "__main__":
    main() 