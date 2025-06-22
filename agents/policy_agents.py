"""
CrewAI Agents for Policy Analysis and Question Answering
"""

from crewai import Agent, Task, Crew, Process
from typing import List, Dict, Optional
from loguru import logger
import sys
from pathlib import Path

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CREWAI_CONFIG
from core.llm_interface import OllamaInterface
from core.vector_store import VectorStore


class PolicyAgents:
    """CrewAI agents for policy analysis and question answering."""
    
    def __init__(self, vector_store: VectorStore, llm_interface: OllamaInterface):
        """Initialize the policy agents."""
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.config = CREWAI_CONFIG
        
        # Initialize agents
        self._create_agents()
    
    def _create_agents(self):
        """Create the specialized agents."""
        
        # Research Agent - Finds relevant policy documents
        self.research_agent = Agent(
            role="Policy Research Specialist",
            goal="Find the most relevant policy documents and information to answer user questions",
            backstory="""You are an expert at analyzing policy documents and finding relevant information. 
            You have years of experience in HR, legal, and corporate policy analysis. You can quickly 
            identify which policies are relevant to specific questions and extract the most important 
            information from them.""",
            verbose=self.config["verbose"],
            allow_delegation=False,
            tools=[self._search_policies_tool],
            llm=self.llm_interface
        )
        
        # Analysis Agent - Analyzes and interprets policy information
        self.analysis_agent = Agent(
            role="Policy Analysis Expert",
            goal="Analyze policy information and provide clear, accurate interpretations",
            backstory="""You are a senior policy analyst with expertise in interpreting complex policy 
            documents. You can break down complicated policies into clear, understandable explanations 
            and identify the key points that are most relevant to specific situations. You have a 
            strong background in legal interpretation and corporate compliance.""",
            verbose=self.config["verbose"],
            allow_delegation=False,
            tools=[self._analyze_policy_tool],
            llm=self.llm_interface
        )
        
        # Communication Agent - Provides clear, user-friendly responses
        self.communication_agent = Agent(
            role="Policy Communication Specialist",
            goal="Provide clear, helpful, and professional responses to policy questions",
            backstory="""You are an expert communicator who specializes in explaining complex policy 
            information in a clear, user-friendly way. You have excellent writing skills and can 
            present information in a professional yet approachable manner. You always ensure that 
            responses are accurate, complete, and easy to understand.""",
            verbose=self.config["verbose"],
            allow_delegation=False,
            tools=[self._format_response_tool],
            llm=self.llm_interface
        )
    
    def _search_policies_tool(self, query: str) -> str:
        """Tool for searching policy documents."""
        try:
            results = self.vector_store.search(query, top_k=5)
            
            if not results:
                return "No relevant policy documents found for this query."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"Document {i}:\n"
                    f"Content: {result['document']}\n"
                    f"Relevance Score: {result['similarity']:.3f}\n"
                    f"Metadata: {result['metadata']}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error in search_policies_tool: {e}")
            return f"Error searching policies: {str(e)}"
    
    def _analyze_policy_tool(self, policy_text: str) -> str:
        """Tool for analyzing policy text."""
        try:
            # Use the policy assistant to analyze the text
            from core.llm_interface import PolicyAssistant
            assistant = PolicyAssistant(self.llm_interface)
            
            # Extract key points
            key_points = assistant.extract_key_points(policy_text)
            
            # Create analysis
            analysis = f"""Policy Analysis:

Key Points:
{chr(10).join(f"- {point}" for point in key_points)}

Summary:
{assistant.summarize_policy(policy_text)}

This analysis provides the essential information from the policy document in a clear, structured format."""
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_policy_tool: {e}")
            return f"Error analyzing policy: {str(e)}"
    
    def _format_response_tool(self, content: str) -> str:
        """Tool for formatting the final response."""
        try:
            # Use the LLM to format the response professionally
            prompt = f"""Please format the following policy information into a clear, professional response:

{content}

Please ensure the response is:
1. Well-structured and easy to read
2. Professional in tone
3. Accurate and complete
4. Helpful to the user

Formatted Response:"""
            
            response = self.llm_interface.generate_response(prompt, max_tokens=1000)
            return response
            
        except Exception as e:
            logger.error(f"Error in format_response_tool: {e}")
            return content  # Return original content if formatting fails
    
    def answer_question(self, question: str) -> Dict[str, any]:
        """
        Answer a policy question using the agent crew.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Create tasks
            research_task = Task(
                description=f"""Research and find the most relevant policy documents for this question: "{question}"
                
                Your task is to:
                1. Search for policy documents that are relevant to the question
                2. Identify the most important and relevant information
                3. Provide a summary of what you found
                
                Use the search_policies_tool to find relevant documents.""",
                agent=self.research_agent,
                expected_output="A comprehensive list of relevant policy documents and key information found"
            )
            
            analysis_task = Task(
                description=f"""Analyze the policy information found by the research agent for this question: "{question}"
                
                Your task is to:
                1. Review the policy documents and information provided
                2. Analyze the relevance and importance of each piece of information
                3. Identify the key points that directly answer the question
                4. Provide a detailed analysis of the policy implications
                
                Use the analyze_policy_tool to help with your analysis.""",
                agent=self.analysis_agent,
                expected_output="A detailed analysis of the policy information and its relevance to the question",
                context=[research_task]
            )
            
            communication_task = Task(
                description=f"""Create a clear, professional response to this question: "{question}"
                
                Your task is to:
                1. Take the analysis provided and create a comprehensive answer
                2. Ensure the response is clear, accurate, and helpful
                3. Format the response in a professional manner
                4. Include relevant policy references and citations
                
                Use the format_response_tool to help create the final response.""",
                agent=self.communication_agent,
                expected_output="A clear, professional, and comprehensive answer to the user's question",
                context=[analysis_task]
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[self.research_agent, self.analysis_agent, self.communication_agent],
                tasks=[research_task, analysis_task, communication_task],
                process=Process.sequential,
                verbose=self.config["verbose"],
                max_iterations=self.config["max_iterations"]
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            return {
                "answer": result,
                "question": question,
                "agents_used": ["research", "analysis", "communication"],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "question": question,
                "agents_used": [],
                "status": "error",
                "error": str(e)
            }
    
    def analyze_policy_document(self, document_text: str) -> Dict[str, any]:
        """
        Analyze a specific policy document.
        
        Args:
            document_text: The policy document text
            
        Returns:
            Dictionary containing the analysis
        """
        try:
            # Create analysis task
            analysis_task = Task(
                description=f"""Analyze the following policy document and provide a comprehensive analysis:

Document:
{document_text}

Your task is to:
1. Identify the key points and requirements
2. Summarize the main policy provisions
3. Highlight important deadlines, procedures, or requirements
4. Note any exceptions or special conditions
5. Provide a clear, structured analysis

Use the analyze_policy_tool to help with your analysis.""",
                agent=self.analysis_agent,
                expected_output="A comprehensive analysis of the policy document including key points, summary, and important details"
            )
            
            # Create formatting task
            formatting_task = Task(
                description="Format the policy analysis into a clear, professional document that is easy to read and understand.",
                agent=self.communication_agent,
                expected_output="A well-formatted, professional policy analysis document",
                context=[analysis_task]
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[self.analysis_agent, self.communication_agent],
                tasks=[analysis_task, formatting_task],
                process=Process.sequential,
                verbose=self.config["verbose"]
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            return {
                "analysis": result,
                "document_length": len(document_text),
                "agents_used": ["analysis", "communication"],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_policy_document: {e}")
            return {
                "analysis": f"Error analyzing policy document: {str(e)}",
                "document_length": len(document_text),
                "agents_used": [],
                "status": "error",
                "error": str(e)
            }
    
    def get_policy_recommendations(self, question: str, context: str = None) -> Dict[str, any]:
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
                
                Additional context: {context or "None provided"}""",
                agent=self.analysis_agent,
                expected_output="Specific policy recommendations and actionable advice",
                context=[research_task]
            )
            
            # Create formatting task
            formatting_task = Task(
                description="Format the recommendations into a clear, actionable document.",
                agent=self.communication_agent,
                expected_output="A well-formatted document with clear recommendations",
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
    """Test the policy agents."""
    try:
        # Initialize components
        vector_store = VectorStore()
        llm_interface = OllamaInterface()
        agents = PolicyAgents(vector_store, llm_interface)
        
        # Test question answering
        question = "What is the vacation policy for employees?"
        result = agents.answer_question(question)
        
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Status: {result['status']}")
        
    except Exception as e:
        print(f"Error testing policy agents: {e}")


if __name__ == "__main__":
    main() 