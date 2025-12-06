"""
Transformer-based AI Text Detector

This module uses HuggingFace transformers for AI text detection.
Supports multiple pre-trained detector models.
"""

import numpy as np
from typing import Dict, List, Optional


class TransformerDetector:
    """
    Transformer-based classifier for AI text detection.
    Uses pre-trained models for inference.
    """
    
    # Pre-trained model options for AI detection
    AVAILABLE_MODELS = {
        'roberta-openai': 'openai-community/roberta-base-openai-detector',
        'roberta-large-openai': 'openai-community/roberta-large-openai-detector', 
        'hello-simple-ai': 'Hello-SimpleAI/chatgpt-detector-roberta',
        'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english',
    }
    
    def __init__(self, model_name: str = 'roberta-openai', device: Optional[str] = None):
        """
        Initialize the transformer detector.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_key = model_name
        self.model_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.device = device
        self.load_error = None
        
    def load_model(self) -> bool:
        """Load the model and tokenizer."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            print(f"Loading model: {self.model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            self.load_error = str(e)
            print(f"Error loading model {self.model_name}: {e}")
            
            # Try fallback model
            if self.model_key != 'distilbert':
                print("Trying fallback model...")
                try:
                    import torch
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    
                    fallback = 'distilbert-base-uncased-finetuned-sst-2-english'
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback)
                    self.model = AutoModelForSequenceClassification.from_pretrained(fallback)
                    self.model.to(self.device)
                    self.model.eval()
                    self.model_name = fallback
                    self.is_loaded = True
                    print(f"Fallback model loaded: {fallback}")
                    return True
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
            
            self.is_loaded = False
            return False
    
    def predict(self, text: str, max_length: int = 512) -> Dict[str, float]:
        """
        Predict if text is AI-generated.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    'label': 0,
                    'label_str': 'Unknown',
                    'ai_probability': 0.5,
                    'human_probability': 0.5,
                    'confidence': 0.0,
                    'error': self.load_error or 'Model not loaded',
                    'model': self.model_name
                }
        
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Get label names from model config
            id2label = getattr(self.model.config, 'id2label', None)
            
            # Determine which index corresponds to AI/Fake
            # For OpenAI detector: label 0 = Real, label 1 = Fake
            # For sentiment models: we'll use positive as AI proxy
            if id2label:
                labels = [id2label.get(i, str(i)).lower() for i in range(len(probs))]
                
                # Find fake/ai index
                fake_idx = None
                for i, label in enumerate(labels):
                    if any(x in label for x in ['fake', 'ai', 'generated', 'positive', 'chatgpt']):
                        fake_idx = i
                        break
                
                if fake_idx is not None:
                    ai_prob = float(probs[fake_idx])
                    human_prob = 1 - ai_prob
                else:
                    # Default: assume index 1 is AI
                    ai_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    human_prob = 1 - ai_prob
            else:
                # Default: index 1 is AI/Fake
                ai_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                human_prob = 1 - ai_prob
            
            label = 1 if ai_prob >= 0.5 else 0
            
            return {
                'label': label,
                'label_str': 'AI-generated' if label == 1 else 'Human-written',
                'ai_probability': ai_prob,
                'human_probability': human_prob,
                'confidence': float(max(ai_prob, human_prob)),
                'model': self.model_name
            }
            
        except Exception as e:
            return {
                'label': 0,
                'label_str': 'Error',
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'confidence': 0.0,
                'error': str(e),
                'model': self.model_name
            }
    
    def predict_batch(self, texts: List[str], max_length: int = 512, 
                      batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Predict for multiple texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                results.append(self.predict(text, max_length))
        
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_key': self.model_key,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'error': self.load_error
        }
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Get list of available models."""
        return TransformerDetector.AVAILABLE_MODELS.copy()


class MockTransformerDetector:
    """
    Mock transformer detector for testing without loading actual models.
    Uses simple heuristics to simulate transformer behavior.
    """
    
    def __init__(self, *args, **kwargs):
        self.is_loaded = True
        self.model_name = 'mock-detector'
        self.model_key = 'mock'
        self.device = 'cpu'
        self.load_error = None
        
    def load_model(self) -> bool:
        return True
    
    def predict(self, text: str, **kwargs) -> Dict[str, float]:
        """
        Simple heuristic-based prediction.
        Uses multiple signals to estimate AI likelihood.
        """
        score = 0.5
        
        # Formal/academic indicators (AI-like)
        formal_words = [
            'therefore', 'however', 'furthermore', 'moreover', 
            'consequently', 'additionally', 'specifically',
            'represents', 'demonstrates', 'indicates', 'significant',
            'comprehensive', 'fundamental', 'implementation',
            '因此', '然而', '此外', '同時', '進而', '顯著', '表明',
            '實現', '構成', '具有', '促進', '推動', '提升'
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_words if word in text_lower or word in text)
        score += min(formal_count * 0.03, 0.25)
        
        # Casual/informal indicators (Human-like)
        casual_markers = [
            'lol', 'haha', 'omg', '!!', '??', 'tbh', 'ngl', 'fr',
            '哈哈', 'QQ', '欸', '啦', '喔', '～', '...', '!!'
        ]
        casual_count = sum(1 for marker in casual_markers if marker in text or marker in text_lower)
        score -= min(casual_count * 0.05, 0.25)
        
        # Sentence structure consistency (AI tends to be more uniform)
        sentences = [s.strip() for s in text.replace('。', '.').split('.') if s.strip()]
        if len(sentences) > 2:
            lengths = [len(s.split()) for s in sentences]
            if lengths:
                variance = np.var(lengths)
                if variance < 5:  # Low variance = more AI-like
                    score += 0.1
                elif variance > 20:  # High variance = more human-like
                    score -= 0.1
        
        # Structural markers (AI-like)
        if any(marker in text for marker in ['First,', 'Second,', 'Third,', 
                                              'In conclusion', 'To summarize',
                                              '首先', '其次', '最後', '總結']):
            score += 0.1
        
        # Personal pronouns and emotional expressions (Human-like)
        personal_markers = ['i ', "i'm", "i've", 'my ', 'me ', 
                           '我', '我們', '覺得', '好想', '超']
        personal_count = sum(1 for marker in personal_markers if marker in text_lower or marker in text)
        score -= min(personal_count * 0.02, 0.15)
        
        # Cap score
        score = max(min(score, 0.95), 0.05)
        label = 1 if score >= 0.5 else 0
        
        return {
            'label': label,
            'label_str': 'AI-generated' if label == 1 else 'Human-written',
            'ai_probability': score,
            'human_probability': 1 - score,
            'confidence': abs(score - 0.5) * 2,
            'model': self.model_name
        }
    
    def predict_batch(self, texts: List[str], **kwargs) -> List[Dict[str, float]]:
        return [self.predict(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, str]:
        return {
            'model_name': self.model_name,
            'model_key': self.model_key,
            'device': 'cpu',
            'is_loaded': True,
            'error': None
        }


def get_detector(use_mock: bool = False, model_name: str = 'roberta-openai') -> TransformerDetector:
    """
    Factory function to get appropriate detector.
    
    Args:
        use_mock: If True, return mock detector for testing
        model_name: Model to use if not mock
        
    Returns:
        Detector instance
    """
    if use_mock:
        return MockTransformerDetector()
    return TransformerDetector(model_name=model_name)
