"""
Free LLM Interface for Groq and Hugging Face models with Bengali support
"""

import os
from typing import List, Dict, Optional, Any
from loguru import logger
import sys
import json
from pathlib import Path
import re

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import FREE_LLM_CONFIG, GROQ_API_KEY, LLM_PROVIDER

try:
    import groq
except ImportError:
    groq = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except ImportError:
    transformers = None
    torch = None


class FreeLLMInterface:
    """Interface for free LLM services (Groq and Hugging Face) with Bengali support."""
    
    def __init__(self, provider: str = "groq", model: str = None):
        """
        Initialize the free LLM interface.
        
        Args:
            provider: "groq" or "huggingface"
            model: Model name/ID
        """
        self.provider = provider
        self.config = FREE_LLM_CONFIG.get(provider, {})
        
        if provider == "groq":
            self._init_groq(model)
        elif provider == "huggingface":
            self._init_huggingface(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_groq(self, model: str = None):
        """Initialize Groq client."""
        if groq is None:
            raise ImportError("Groq package not installed. Run: pip install groq")
        
        # Get API key from environment
        api_key = GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
        
        self.client = groq.Groq(api_key=api_key)
        self.model = model or self.config.get("default_model", "llama3-8b-8192")
        
        logger.info(f"Initialized Groq with model: {self.model}")
    
    def _init_huggingface(self, model: str = None):
        """Initialize Hugging Face model."""
        if transformers is None or torch is None:
            raise ImportError("Transformers and torch packages not installed")
        
        self.model_name = model or self.config.get("default_model", "microsoft/DialoGPT-medium")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"Initialized Hugging Face with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text contains Bengali characters.
        
        Args:
            text: Text to analyze
            
        Returns:
            'bengali' if Bengali detected, 'english' otherwise
        """
        if not text:
            return 'english'
        
        # Bengali character patterns
        bengali_patterns = [
            r'[অ-ঙ]',  # Bengali consonants
            r'[ক-হ]',  # Bengali consonants
            r'[ড়-ঢ়]',  # Bengali consonants
            r'[য়-ৎ]',  # Bengali consonants and symbols
            r'[া-ৌ]',  # Bengali vowel signs
            r'[্]',    # Bengali halant
            r'[০-৯]',  # Bengali numerals
        ]
        
        # Count Bengali characters
        bengali_count = 0
        total_chars = len(text)
        
        for pattern in bengali_patterns:
            matches = re.findall(pattern, text)
            bengali_count += len(matches)
        
        # If more than 10% of characters are Bengali, consider it Bengali
        if total_chars > 0 and (bengali_count / total_chars) > 0.1:
            return 'bengali'
        
        return 'english'
    
    def generate_response(self, prompt: str, context: str = None, max_tokens: int = 1000) -> str:
        """
        Generate a response using the selected LLM.
        
        Args:
            prompt: The main prompt/question
            context: Optional context information
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Detect language
            language = self.detect_language(prompt)
            
            if self.provider == "groq":
                return self._generate_groq(prompt, context, max_tokens, language)
            elif self.provider == "huggingface":
                return self._generate_huggingface(prompt, context, max_tokens, language)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def _generate_groq(self, prompt: str, context: str = None, max_tokens: int = 1000, language: str = "english") -> str:
        """Generate response using Groq."""
        # Build the full prompt with context
        if context:
            if language == "bengali":
                full_prompt = f"""প্রসঙ্গ: {context}

প্রশ্ন: {prompt}

অনুগ্রহ করে প্রদত্ত প্রসঙ্গের উপর ভিত্তি করে একটি স্পষ্ট এবং সঠিক উত্তর দিন। যদি প্রসঙ্গে প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য না থাকে, অনুগ্রহ করে তা বলুন।"""
            else:
                full_prompt = f"""Context: {context}

Question: {prompt}

Please provide a clear and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""
        else:
            full_prompt = prompt
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about company policies accurately and professionally. You can respond in both English and Bengali."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_huggingface(self, prompt: str, context: str = None, max_tokens: int = 1000, language: str = "english") -> str:
        """Generate response using Hugging Face model."""
        # Build the full prompt
        if context:
            if language == "bengali":
                full_prompt = f"প্রসঙ্গ: {context}\n\nপ্রশ্ন: {prompt}\n\nউত্তর:"
            else:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            if language == "bengali":
                full_prompt = f"প্রশ্ন: {prompt}\n\nউত্তর:"
            else:
                full_prompt = f"Question: {prompt}\n\nAnswer:"
        
        # Generate response
        response = self.pipeline(
            full_prompt,
            max_length=len(self.tokenizer.encode(full_prompt)) + max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract the generated text
        generated_text = response[0]['generated_text']
        
        # Remove the input prompt to get only the response
        response_text = generated_text[len(full_prompt):].strip()
        
        return response_text
    
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
            # Detect language
            language = self.detect_language(prompt)
            
            # Add format instruction to prompt
            if response_format == "json":
                if language == "bengali":
                    format_instruction = "অনুগ্রহ করে বৈধ JSON ফরম্যাটে উত্তর দিন।"
                else:
                    format_instruction = "Please respond in valid JSON format."
            else:
                if language == "bengali":
                    format_instruction = f"অনুগ্রহ করে {response_format} ফরম্যাটে উত্তর দিন।"
                else:
                    format_instruction = f"Please respond in {response_format} format."
            
            if context:
                if language == "bengali":
                    full_prompt = f"""প্রসঙ্গ: {context}

প্রশ্ন: {prompt}

{format_instruction}"""
                else:
                    full_prompt = f"""Context: {context}

Question: {prompt}

{format_instruction}"""
            else:
                if language == "bengali":
                    full_prompt = f"{prompt}\n\n{format_instruction}"
                else:
                    full_prompt = f"{prompt}\n\n{format_instruction}"
            
            # Generate response
            response_text = self.generate_response(full_prompt, max_tokens=1000)
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            if self.provider == "groq":
                return {
                    "provider": "groq",
                    "model": self.model,
                    "type": "cloud_api"
                }
            elif self.provider == "huggingface":
                return {
                    "provider": "huggingface",
                    "model": self.model_name,
                    "device": self.device,
                    "type": "local_model"
                }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}


class PolicyAssistant:
    """Specialized assistant for policy-related queries using free models with Bengali support."""
    
    def __init__(self, llm_interface: FreeLLMInterface):
        """Initialize the policy assistant."""
        self.llm = llm_interface
        self.system_prompt = """You are a helpful assistant that answers questions about company policies, SOPs, and HR procedures. 
        
You can respond in both English and Bengali based on the user's language preference.

Your responses should be:
1. Accurate and based on the provided context
2. Clear and easy to understand
3. Professional and helpful
4. Specific to the company's policies
5. In the same language as the user's question

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
            
            # Detect language
            language = self.llm.detect_language(question)
            
            # Create the prompt
            if language == "bengali":
                prompt = f"""নিম্নলিখিত নীতি নথিগুলির উপর ভিত্তি করে, অনুগ্রহ করে এই প্রশ্নের উত্তর দিন:

প্রশ্ন: {question}

নীতি নথি:
{context}

অনুগ্রহ করে নীতি নথি থেকে তথ্য ব্যবহার করে প্রশ্নের একটি বিস্তৃত উত্তর দিন।"""
            else:
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
            # Detect language
            language = self.llm.detect_language(policy_text)
            
            if language == "bengali":
                prompt = f"""অনুগ্রহ করে নিম্নলিখিত নীতি নথির একটি স্পষ্ট এবং সংক্ষিপ্ত সারসংক্ষেপ প্রদান করুন:

{policy_text}

সারসংক্ষেপ:"""
            else:
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
            # Detect language
            language = self.llm.detect_language(policy_text)
            
            if language == "bengali":
                prompt = f"""অনুগ্রহ করে নিম্নলিখিত নীতি নথি থেকে মূল বিষয়গুলি বের করুন। এগুলি একটি সংখ্যাযুক্ত তালিকা হিসাবে ফিরিয়ে দিন:

{policy_text}

মূল বিষয়গুলি:"""
            else:
                prompt = f"""Please extract the key points from the following policy document. Return them as a numbered list:

{policy_text}

Key Points:"""
            
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
    """Test the free LLM interface with Bengali support."""
    try:
        # Test with Groq (if API key is available)
        print("Testing Groq...")
        groq_llm = FreeLLMInterface("groq")
        
        # Test English
        response = groq_llm.generate_response("What is 2+2?")
        print(f"Groq English response: {response}")
        
        # Test Bengali
        bengali_response = groq_llm.generate_response("দুই যোগ দুই কত?")
        print(f"Groq Bengali response: {bengali_response}")
        
        # Test with Hugging Face
        print("Testing Hugging Face...")
        hf_llm = FreeLLMInterface("huggingface")
        
        # Test English
        response = hf_llm.generate_response("What is 2+2?")
        print(f"Hugging Face English response: {response}")
        
        # Test Bengali
        bengali_response = hf_llm.generate_response("দুই যোগ দুই কত?")
        print(f"Hugging Face Bengali response: {bengali_response}")
        
        # Test policy assistant
        assistant = PolicyAssistant(hf_llm)
        policy_text = "Employees are entitled to 20 vacation days per year. Vacation requests must be submitted at least 2 weeks in advance."
        summary = assistant.summarize_policy(policy_text)
        print(f"Policy summary: {summary}")
        
        # Test Bengali policy
        bengali_policy = "কর্মচারীরা বছরে ২০ দিন ছুটি পাবেন। ছুটির আবেদন কমপক্ষে ২ সপ্তাহ আগে জমা দিতে হবে।"
        bengali_summary = assistant.summarize_policy(bengali_policy)
        print(f"Bengali policy summary: {bengali_summary}")
        
    except Exception as e:
        print(f"Error testing free LLM interface: {e}")


if __name__ == "__main__":
    main() 