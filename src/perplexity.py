"""
Perplexity Calculator for AI/Human Text Detection

This module calculates text perplexity using small language models.
Lower perplexity often indicates more predictable (AI-like) text.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class PerplexityCalculator:
    """
    Calculate perplexity using various methods.
    Supports both statistical n-gram models and neural models.
    """
    
    def __init__(self, use_neural: bool = False):
        """
        Initialize the perplexity calculator.
        
        Args:
            use_neural: If True, try to use neural language model
        """
        self.use_neural = use_neural
        self.neural_model = None
        self.neural_tokenizer = None
        self.ngram_model = {}
        self.vocab = set()
        
        if use_neural:
            self._load_neural_model()
    
    def _load_neural_model(self):
        """Load a small neural language model for perplexity calculation."""
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            # Use distilgpt2 for efficiency
            model_name = "distilgpt2"
            self.neural_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.neural_model = GPT2LMHeadModel.from_pretrained(model_name)
            self.neural_model.eval()
            
            # Move to GPU if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.neural_model.to(self.device)
            
            print(f"Loaded neural model: {model_name} on {self.device}")
            
        except Exception as e:
            print(f"Could not load neural model: {e}")
            self.neural_model = None
            self.neural_tokenizer = None
    
    def build_ngram_model(self, texts: List[str], n: int = 3):
        """
        Build an n-gram language model from training texts.
        
        Args:
            texts: List of training texts
            n: N-gram order (default trigram)
        """
        self.n = n
        ngram_counts = Counter()
        context_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            self.vocab.update(tokens)
            
            # Add padding
            padded = ['<s>'] * (n - 1) + tokens + ['</s>']
            
            for i in range(len(padded) - n + 1):
                ngram = tuple(padded[i:i+n])
                context = ngram[:-1]
                
                ngram_counts[ngram] += 1
                context_counts[context] += 1
        
        # Calculate probabilities with add-k smoothing
        k = 0.01
        vocab_size = len(self.vocab) + 2  # +2 for <s> and </s>
        
        self.ngram_model = {}
        for ngram, count in ngram_counts.items():
            context = ngram[:-1]
            prob = (count + k) / (context_counts[context] + k * vocab_size)
            self.ngram_model[ngram] = prob
        
        self.context_counts = context_counts
        self.vocab_size = vocab_size
        self.k = k
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Handle both English and Chinese
        # For Chinese, use character-level tokenization
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if len(chinese_chars) > len(english_words):
            # Chinese text - use characters
            return list(re.sub(r'[^\u4e00-\u9fff]', '', text))
        else:
            # English text - use words
            return english_words
    
    def calculate_ngram_perplexity(self, text: str) -> float:
        """
        Calculate perplexity using n-gram model.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score (lower = more predictable)
        """
        if not self.ngram_model:
            # Use simple character-level entropy as fallback
            return self._calculate_simple_perplexity(text)
        
        tokens = self._tokenize(text)
        if len(tokens) < self.n:
            return self._calculate_simple_perplexity(text)
        
        padded = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        
        log_prob_sum = 0
        count = 0
        
        for i in range(len(padded) - self.n + 1):
            ngram = tuple(padded[i:i+self.n])
            context = ngram[:-1]
            
            if ngram in self.ngram_model:
                prob = self.ngram_model[ngram]
            else:
                # Smoothed probability for unseen n-grams
                context_count = self.context_counts.get(context, 0)
                prob = self.k / (context_count + self.k * self.vocab_size)
            
            log_prob_sum += math.log2(prob)
            count += 1
        
        if count == 0:
            return 100.0  # Default high perplexity
        
        avg_log_prob = log_prob_sum / count
        perplexity = 2 ** (-avg_log_prob)
        
        return min(perplexity, 10000)  # Cap at 10000
    
    def _calculate_simple_perplexity(self, text: str) -> float:
        """
        Calculate approximate perplexity using character entropy.
        
        Args:
            text: Input text
            
        Returns:
            Approximate perplexity
        """
        if not text:
            return 100.0
        
        # Character-level entropy
        char_counts = Counter(text.lower())
        total = len(text)
        
        entropy = 0
        for count in char_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        # Convert entropy to perplexity
        perplexity = 2 ** entropy
        
        return perplexity
    
    def calculate_neural_perplexity(self, text: str, max_length: int = 512) -> float:
        """
        Calculate perplexity using neural language model.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Perplexity score
        """
        if self.neural_model is None or self.neural_tokenizer is None:
            return self._calculate_simple_perplexity(text)
        
        try:
            import torch
            
            # Tokenize
            encodings = self.neural_tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.neural_model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
            
            # Convert loss to perplexity
            perplexity = math.exp(loss)
            
            return min(perplexity, 10000)  # Cap at 10000
            
        except Exception as e:
            print(f"Error calculating neural perplexity: {e}")
            return self._calculate_simple_perplexity(text)
    
    def calculate_perplexity(self, text: str) -> Dict[str, float]:
        """
        Calculate perplexity using available methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with perplexity scores
        """
        results = {
            'simple_perplexity': self._calculate_simple_perplexity(text)
        }
        
        if self.ngram_model:
            results['ngram_perplexity'] = self.calculate_ngram_perplexity(text)
        
        if self.neural_model is not None:
            results['neural_perplexity'] = self.calculate_neural_perplexity(text)
        
        # Combined score (average of available methods)
        scores = list(results.values())
        results['combined_perplexity'] = np.mean(scores)
        
        return results
    
    def get_perplexity_features(self, text: str) -> Dict[str, float]:
        """
        Extract perplexity-based features for classification.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of perplexity features
        """
        perplexity = self.calculate_perplexity(text)
        
        # Normalize perplexity to 0-1 range using sigmoid-like function
        def normalize(ppl: float, scale: float = 100) -> float:
            return 1 / (1 + math.exp(-(ppl - scale) / (scale / 2)))
        
        return {
            'perplexity_score': perplexity.get('combined_perplexity', 100),
            'perplexity_normalized': normalize(perplexity.get('combined_perplexity', 100)),
            'char_perplexity': perplexity.get('simple_perplexity', 100)
        }


class BurstinessCalculator:
    """
    Calculate burstiness - a measure of word usage patterns.
    AI text tends to have more uniform word distributions (low burstiness).
    """
    
    def __init__(self):
        pass
    
    def calculate_burstiness(self, text: str) -> float:
        """
        Calculate burstiness score.
        
        Burstiness measures how "bursty" word occurrences are.
        Human text tends to be more bursty (words cluster together).
        AI text tends to be more uniform (low burstiness).
        
        Args:
            text: Input text
            
        Returns:
            Burstiness score (-1 to 1, higher = more bursty/human-like)
        """
        import re
        
        # Tokenize
        words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]+\b', text.lower())
        
        if len(words) < 10:
            return 0.0
        
        # Calculate inter-arrival times for each word
        word_positions = {}
        for i, word in enumerate(words):
            if word not in word_positions:
                word_positions[word] = []
            word_positions[word].append(i)
        
        # Calculate burstiness for words that appear multiple times
        burstiness_scores = []
        
        for word, positions in word_positions.items():
            if len(positions) >= 2:
                # Calculate inter-arrival times
                intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                
                if intervals:
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    
                    if mean_interval > 0:
                        # Burstiness formula: (std - mean) / (std + mean)
                        burstiness = (std_interval - mean_interval) / (std_interval + mean_interval + 1e-10)
                        burstiness_scores.append(burstiness)
        
        if burstiness_scores:
            return np.mean(burstiness_scores)
        return 0.0
    
    def get_burstiness_features(self, text: str) -> Dict[str, float]:
        """
        Extract burstiness-based features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of burstiness features
        """
        burstiness = self.calculate_burstiness(text)
        
        return {
            'burstiness': burstiness,
            'burstiness_normalized': (burstiness + 1) / 2  # Normalize to 0-1
        }


def get_advanced_features(text: str, perplexity_calc: Optional[PerplexityCalculator] = None) -> Dict[str, float]:
    """
    Get all advanced linguistic features including perplexity and burstiness.
    
    Args:
        text: Input text
        perplexity_calc: Optional pre-initialized perplexity calculator
        
    Returns:
        Dictionary of advanced features
    """
    if perplexity_calc is None:
        perplexity_calc = PerplexityCalculator(use_neural=False)
    
    burstiness_calc = BurstinessCalculator()
    
    features = {}
    features.update(perplexity_calc.get_perplexity_features(text))
    features.update(burstiness_calc.get_burstiness_features(text))
    
    return features
