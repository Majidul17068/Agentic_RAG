"""
LLM Interface for Ollama integration
"""

import ollama
from typing import List, Dict, Optional, Any
from loguru import logger
import sys
import json
from pathlib import Path

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import OLLAMA_CONFIG


class OllamaInterface:
    """Interface for Ollama LLM operations."""
    
    def __init__(self, model: str = None, base_url: str = None):
        """Initialize the Ollama interface."""
        self.model = model or OLLAMA_CONFIG["model"]
        self.base_url = base_url or OLLAMA_CONFIG["base_url"]
        self.timeout = OLLAMA_CONFIG["timeout"]
        
        # Configure ollama client
        ollama.set_host(self.base_url)
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama connection and model availability."""
        try:
            # Check if model is available
            models = ollama.list()
            model_names = [model["name"] for model in models["models"]]
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                logger.info("Attempting to pull the model...")
                ollama.pull(self.model)
                logger.info(f"Successfully pulled model: {self.model}")
            else:
                logger.info(f"Model {self.model} is available")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
    
    def generate_response(self, prompt: str, context: str = None, max_tokens: int = 1000) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The main prompt/question
            context: Optional context information
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Build the full prompt with context
            if context:
                full_prompt = f"""Context: {context}

Question: {prompt}

Please provide a clear and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""
            else:
                full_prompt = prompt
            
            # Generate response
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            
            return response["response"].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def generate_structured_response(self, prompt: str, context: str = None, 
                                   response_format: str = "json") -> Dict[str, Any]:
        """
        Generate a structured response (e.g., JSON).
        
        Args:
            prompt: The main prompt/question
            context: Optional context information
            response_format: Expected response format
            
        Returns:
            Structured response as dictionary
        """
        try:
            # Add format instruction to prompt
            if response_format == "json":
                format_instruction = "Please respond in valid JSON format."
            else:
                format_instruction = f"Please respond in {response_format} format."
            
            if context:
                full_prompt = f"""Context: {context}

Question: {prompt}

{format_instruction}"""
            else:
                full_prompt = f"{prompt}\n\n{format_instruction}"
            
            # Generate response
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "num_predict": 1000,
                    "temperature": 0.3,  # Lower temperature for structured output
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            
            response_text = response["response"].strip()
            
            # Try to parse JSON if requested
            if response_format == "json":
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response, returning as text")
                    return {"response": response_text, "format": "text"}
            
            return {"response": response_text, "format": response_format}
            
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            return {"error": str(e)}
    
    def chat_conversation(self, messages: List[Dict[str, str]], 
                         system_prompt: str = None) -> str:
        """
        Have a conversation with the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt
            
        Returns:
            LLM response
        """
        try:
            # Prepare messages for Ollama
            ollama_messages = []
            
            if system_prompt:
                ollama_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            ollama_messages.extend(messages)
            
            # Generate response
            response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                options={
                    "num_predict": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Error in chat conversation: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            models = ollama.list()
            for model in models["models"]:
                if model["name"] == self.model:
                    return {
                        "name": model["name"],
                        "size": model.get("size", "unknown"),
                        "modified_at": model.get("modified_at", "unknown"),
                        "digest": model.get("digest", "unknown")
                    }
            return {"error": f"Model {self.model} not found"}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def stream_response(self, prompt: str, context: str = None):
        """
        Stream response tokens as they're generated.
        
        Args:
            prompt: The main prompt/question
            context: Optional context information
            
        Yields:
            Response tokens as they're generated
        """
        try:
            if context:
                full_prompt = f"""Context: {context}

Question: {prompt}

Please provide a clear and accurate answer based on the context provided."""
            else:
                full_prompt = prompt
            
            # Stream response
            for chunk in ollama.generate(
                model=self.model,
                prompt=full_prompt,
                stream=True,
                options={
                    "num_predict": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            ):
                if "response" in chunk:
                    yield chunk["response"]
                    
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"Error: {str(e)}"


class PolicyAssistant:
    """Specialized assistant for policy-related queries."""
    
    def __init__(self, llm_interface: OllamaInterface):
        """Initialize the policy assistant."""
        self.llm = llm_interface
        self.system_prompt = """You are a helpful assistant that answers questions about company policies, SOPs, and HR procedures. 
        
Your responses should be:
1. Accurate and based on the provided context
2. Clear and easy to understand
3. Professional and helpful
4. Specific to the company's policies

If the context doesn't contain enough information to answer a question, please say so and suggest where the user might find more information."""
    
    def answer_policy_question(self, question: str, context_documents: List[str]) -> str:
        """
        Answer a policy-related question using provided context.
        
        Args:
            question: The user's question
            context_documents: List of relevant policy documents
            
        Returns:
            Answer to the question
        """
        try:
            # Combine context documents
            context = "\n\n".join(context_documents)
            
            # Create the prompt
            prompt = f"""Based on the following policy documents, please answer this question:

Question: {question}

Policy Documents:
{context}

Please provide a comprehensive answer that directly addresses the question using the information from the policy documents."""
            
            # Generate response
            response = self.llm.generate_response(prompt, max_tokens=1500)
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering policy question: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def summarize_policy(self, policy_text: str) -> str:
        """
        Summarize a policy document.
        
        Args:
            policy_text: The policy text to summarize
            
        Returns:
            Summary of the policy
        """
        try:
            prompt = f"""Please provide a clear and concise summary of the following policy document:

{policy_text}

Summary:"""
            
            response = self.llm.generate_response(prompt, max_tokens=500)
            return response
            
        except Exception as e:
            logger.error(f"Error summarizing policy: {e}")
            return f"Error summarizing policy: {str(e)}"
    
    def extract_key_points(self, policy_text: str) -> List[str]:
        """
        Extract key points from a policy document.
        
        Args:
            policy_text: The policy text
            
        Returns:
            List of key points
        """
        try:
            prompt = f"""Please provide a clear and concise summary of the following policy document:

{policy_text}

Summary:"""
            
            response = self.llm.generate_response(prompt, max_tokens=800)
            
            # Parse numbered list
            lines = response.split('\n')
            key_points = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    # Remove numbering/bullets
                    clean_line = line.lstrip('0123456789.-* ').strip()
                    if clean_line:
                        key_points.append(clean_line)
            
            return key_points if key_points else [response]
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return [f"Error extracting key points: {str(e)}"]


def main():
    """Test the LLM interface."""
    try:
        # Initialize interface
        llm = OllamaInterface()
        
        # Test basic response
        response = llm.generate_response("What is 2+2?")
        print(f"Basic response: {response}")
        
        # Test with context
        context = "The company policy states that employees get 20 vacation days per year."
        response = llm.generate_response("How many vacation days do I get?", context)
        print(f"Context response: {response}")
        
        # Test policy assistant
        assistant = PolicyAssistant(llm)
        policy_text = "Employees are entitled to 20 vacation days per year. Vacation requests must be submitted at least 2 weeks in advance."
        summary = assistant.summarize_policy(policy_text)
        print(f"Policy summary: {summary}")
        
    except Exception as e:
        print(f"Error testing LLM interface: {e}")


if __name__ == "__main__":
    main() 