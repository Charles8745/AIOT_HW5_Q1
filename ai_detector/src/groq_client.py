"""
Groq API Client for AI Text Generation

This module provides a wrapper for the Groq API to generate
AI text samples for testing and demonstration purposes.
"""

import os
from typing import Dict, List, Optional


class GroqClient:
    """
    Client for Groq API to generate AI text samples.
    Uses GROQ_API_KEY environment variable for authentication.
    """
    
    AVAILABLE_MODELS = [
        'llama-3.3-70b-versatile',
        'llama-3.1-8b-instant',
        'mixtral-8x7b-32768',
        'gemma2-9b-it'
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'llama-3.1-8b-instant'):
        """
        Initialize the Groq client.
        
        Args:
            api_key: Groq API key. If None, uses GROQ_API_KEY env variable
            model: Model to use for generation
        """
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '')
        self.model = model
        self.client = None
        self.is_available = False
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Groq client if API key is available."""
        if not self.api_key:
            print("Warning: GROQ_API_KEY not set. AI generation will not be available.")
            return
            
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.is_available = True
        except ImportError:
            print("Warning: groq package not installed. Run: pip install groq")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
    
    def generate_text(self, prompt: str, max_tokens: int = 500, 
                      temperature: float = 0.7) -> Dict[str, str]:
        """
        Generate AI text using Groq API.
        
        Args:
            prompt: Prompt for text generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with generated text and metadata
        """
        if not self.is_available:
            return {
                'success': False,
                'text': '',
                'error': 'Groq API not available. Please set GROQ_API_KEY.',
                'model': self.model
            }
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Generate natural, well-written text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                'success': True,
                'text': generated_text,
                'model': self.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'error': str(e),
                'model': self.model
            }
    
    def generate_essay(self, topic: str, style: str = 'academic') -> Dict[str, str]:
        """
        Generate an essay on a given topic.
        
        Args:
            topic: Essay topic
            style: Writing style ('academic', 'casual', 'professional')
            
        Returns:
            Dictionary with generated essay
        """
        style_prompts = {
            'academic': "Write a formal academic essay about",
            'casual': "Write a casual, conversational blog post about",
            'professional': "Write a professional article about"
        }
        
        prompt_prefix = style_prompts.get(style, style_prompts['academic'])
        prompt = f"{prompt_prefix} the following topic: {topic}. Write 2-3 paragraphs."
        
        return self.generate_text(prompt, max_tokens=800)
    
    def generate_samples(self, topics: List[str], n_per_topic: int = 1) -> List[Dict]:
        """
        Generate multiple AI text samples for testing.
        
        Args:
            topics: List of topics to generate text for
            n_per_topic: Number of samples per topic
            
        Returns:
            List of generated samples with metadata
        """
        samples = []
        
        for topic in topics:
            for i in range(n_per_topic):
                result = self.generate_essay(topic)
                if result['success']:
                    samples.append({
                        'topic': topic,
                        'text': result['text'],
                        'model': result['model'],
                        'label': 1  # AI-generated
                    })
        
        return samples
    
    def set_model(self, model: str):
        """Change the model used for generation."""
        if model in self.AVAILABLE_MODELS:
            self.model = model
        else:
            print(f"Warning: Model {model} not in available models. Using {self.model}")
    
    def get_available_models(self) -> List[str]:
        """Return list of available models."""
        return self.AVAILABLE_MODELS.copy()
    
    def is_configured(self) -> bool:
        """Check if the client is properly configured."""
        return self.is_available


class MockGroqClient:
    """
    Mock Groq client for testing without API access.
    Returns pre-defined AI-like text samples.
    """
    
    MOCK_RESPONSES = [
        "Artificial intelligence represents a transformative technology that continues to reshape various aspects of modern society. The integration of AI systems into everyday applications has demonstrated significant potential for improving efficiency and decision-making processes. Furthermore, ongoing research in machine learning and neural networks promises even greater advancements in the coming years.",
        
        "Climate change poses one of the most significant challenges facing humanity in the 21st century. Scientific consensus indicates that human activities, particularly the burning of fossil fuels, have contributed substantially to global warming. Addressing this issue requires coordinated international efforts and significant changes to our energy infrastructure.",
        
        "The digital revolution has fundamentally altered how we communicate, work, and access information. Social media platforms have created new paradigms for human interaction, while e-commerce has transformed traditional retail models. These changes present both opportunities and challenges for individuals and organizations alike.",
        
        "Education systems worldwide are undergoing significant transformation in response to technological advancement and changing workforce demands. The integration of digital tools and online learning platforms has expanded access to educational resources. Additionally, there is growing recognition of the importance of developing critical thinking and adaptability skills.",
        
        "Healthcare innovation continues to advance at a remarkable pace, with new treatments and technologies offering hope for previously untreatable conditions. Precision medicine, enabled by genetic sequencing and data analytics, represents a paradigm shift in how we approach disease prevention and treatment. These developments hold tremendous promise for improving patient outcomes."
    ]
    
    def __init__(self, *args, **kwargs):
        self.is_available = True
        self.model = 'mock-model'
        self._response_index = 0
    
    def generate_text(self, prompt: str, **kwargs) -> Dict[str, str]:
        """Return a mock response."""
        text = self.MOCK_RESPONSES[self._response_index % len(self.MOCK_RESPONSES)]
        self._response_index += 1
        
        return {
            'success': True,
            'text': text,
            'model': self.model,
            'usage': {
                'prompt_tokens': 50,
                'completion_tokens': 150,
                'total_tokens': 200
            }
        }
    
    def generate_essay(self, topic: str, style: str = 'academic') -> Dict[str, str]:
        """Return a mock essay."""
        return self.generate_text(topic)
    
    def generate_samples(self, topics: List[str], n_per_topic: int = 1) -> List[Dict]:
        """Generate mock samples."""
        samples = []
        for topic in topics:
            for _ in range(n_per_topic):
                result = self.generate_text(topic)
                samples.append({
                    'topic': topic,
                    'text': result['text'],
                    'model': self.model,
                    'label': 1
                })
        return samples
    
    def is_configured(self) -> bool:
        return True
    
    def get_available_models(self) -> List[str]:
        return ['mock-model']


def get_groq_client(use_mock: bool = False, **kwargs) -> GroqClient:
    """
    Factory function to get Groq client.
    
    Args:
        use_mock: If True, return mock client for testing
        **kwargs: Additional arguments for client initialization
        
    Returns:
        GroqClient or MockGroqClient instance
    """
    if use_mock:
        return MockGroqClient(**kwargs)
    return GroqClient(**kwargs)
