"""
Feature Engineering Module for AI/Human Text Detection

This module extracts explainable statistical and linguistic features
from text to help distinguish between AI-generated and human-written content.
Supports both English and Chinese text.
"""

import re
import math
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional


class FeatureExtractor:
    """
    Extract statistical and linguistic features from text.
    All features are designed to be explainable and visualizable.
    Supports both English and Chinese text.
    """
    
    # Common English function words
    FUNCTION_WORDS_EN = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
        'now', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would',
        'should', 'could', 'ought', 'i', 'me', 'my', 'myself', 'we', 'our',
        'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it',
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'am', 'will', 'shall', 'may', 'might', 'must', 'can'
    }
    
    # Common Chinese function words (虛詞)
    FUNCTION_WORDS_ZH = {
        '的', '了', '是', '在', '有', '和', '與', '或', '但', '而', '也',
        '都', '就', '要', '會', '可以', '能', '將', '把', '被', '讓', '使',
        '對', '給', '從', '到', '向', '往', '為', '以', '因', '因為', '所以',
        '如果', '雖然', '但是', '然而', '不過', '可是', '而且', '並且', '或者',
        '這', '那', '這個', '那個', '這些', '那些', '什麼', '怎麼', '如何',
        '很', '非常', '太', '更', '最', '比較', '相當', '十分', '極', '挺',
        '我', '你', '他', '她', '它', '我們', '你們', '他們', '她們', '它們',
        '自己', '這裡', '那裡', '哪裡', '什麼時候', '為什麼', '怎麼樣',
        '著', '過', '個', '們', '地', '得', '啊', '呢', '吧', '嗎', '啦', '喔',
        '一', '不', '沒', '沒有', '已經', '正在', '還', '又', '再', '才'
    }

    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = [
            'avg_sentence_length',
            'avg_word_length',
            'sentence_count',
            'word_count',
            'char_count',
            'type_token_ratio',
            'hapax_legomena_ratio',
            'function_word_ratio',
            'punctuation_ratio',
            'uppercase_ratio',
            'digit_ratio',
            'char_entropy',
            'word_entropy',
            'bigram_entropy',
            'trigram_repetition_rate',
            'avg_words_per_sentence',
            'sentence_length_variance',
            'lexical_density',
            'avg_syllables_per_word',
            'long_word_ratio',
            'chinese_char_ratio',
            'formal_word_ratio'
        ]
        
        # Try to import jieba for Chinese tokenization
        self.jieba_available = False
        try:
            import jieba
            jieba.setLogLevel(20)  # Suppress jieba logs
            self.jieba = jieba
            self.jieba_available = True
        except ImportError:
            self.jieba = None

    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Chinese or English.
        
        Args:
            text: Input text
            
        Returns:
            'zh' for Chinese, 'en' for English
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.3 else 'en'

    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text based on language.
        
        Args:
            text: Input text
            language: 'zh' or 'en', auto-detect if None
            
        Returns:
            List of tokens
        """
        if language is None:
            language = self.detect_language(text)
        
        if language == 'zh':
            return self.tokenize_chinese(text)
        else:
            return self.tokenize_english(text)
    
    def tokenize_english(self, text: str) -> List[str]:
        """Tokenize English text."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def tokenize_chinese(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using jieba or character-based fallback.
        """
        if self.jieba_available:
            # Use jieba for word segmentation
            words = list(self.jieba.cut(text))
            # Filter out punctuation and whitespace
            words = [w.strip() for w in words if w.strip() and not re.match(r'^[\s\W]+$', w)]
            return words
        else:
            # Fallback: character-based tokenization for Chinese
            # Extract Chinese characters and English words
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
            english_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            return chinese_chars + english_words

    def get_sentences(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            language: 'zh' or 'en', auto-detect if None
            
        Returns:
            List of sentences
        """
        if language is None:
            language = self.detect_language(text)
        
        if language == 'zh':
            # Chinese sentence delimiters
            sentences = re.split(r'[。！？；\n]+', text)
        else:
            # English sentence delimiters
            sentences = re.split(r'[.!?]+', text)
        
        return [s.strip() for s in sentences if s.strip()]

    def calculate_entropy(self, items: List) -> float:
        """Calculate Shannon entropy of a sequence."""
        if not items:
            return 0.0
        counter = Counter(items)
        total = len(items)
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        return entropy

    def get_ngrams(self, items: List, n: int) -> List[Tuple]:
        """Generate n-grams from a list of items."""
        return [tuple(items[i:i+n]) for i in range(len(items) - n + 1)]

    def count_syllables(self, word: str) -> int:
        """Estimate syllable count in an English word."""
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    def get_function_words(self, language: str) -> set:
        """Get function words for the specified language."""
        if language == 'zh':
            return self.FUNCTION_WORDS_ZH
        return self.FUNCTION_WORDS_EN
    
    def get_formal_indicators(self, text: str, language: str) -> float:
        """
        Calculate ratio of formal/academic indicators in text.
        Higher values suggest more formal/AI-like writing.
        """
        formal_en = [
            'therefore', 'however', 'furthermore', 'moreover', 'consequently',
            'additionally', 'specifically', 'particularly', 'significantly',
            'demonstrates', 'indicates', 'represents', 'constitutes',
            'implementation', 'fundamental', 'comprehensive', 'substantial',
            'unprecedented', 'paradigm', 'framework', 'methodology'
        ]
        
        formal_zh = [
            '因此', '然而', '此外', '同時', '進而', '顯著', '表明', '呈現',
            '實現', '構成', '具有', '促進', '推動', '提升', '優化', '整合',
            '前所未有', '日益', '逐步', '有效', '重要', '關鍵', '核心',
            '深刻', '廣泛', '顯著', '持續', '積極', '充分', '全面'
        ]
        
        text_lower = text.lower()
        words = self.tokenize(text, language)
        
        if language == 'zh':
            formal_count = sum(1 for word in formal_zh if word in text)
            total = len(words) if words else 1
        else:
            formal_count = sum(1 for word in formal_en if word in text_lower)
            total = len(words) if words else 1
        
        return formal_count / total

    def extract_features(self, text: str, language: Optional[str] = None) -> Dict[str, float]:
        """
        Extract all features from a text.
        
        Args:
            text: Input text string
            language: 'zh' or 'en', auto-detect if None
            
        Returns:
            Dictionary of feature names to values
        """
        if not text or not text.strip():
            return {name: 0.0 for name in self.feature_names}
        
        # Detect language if not specified
        if language is None:
            language = self.detect_language(text)

        # Basic tokenization
        words = self.tokenize(text, language)
        sentences = self.get_sentences(text, language)
        chars = list(text)
        
        # Counts
        word_count = len(words)
        sentence_count = len(sentences)
        char_count = len(text)
        
        if word_count == 0:
            return {name: 0.0 for name in self.feature_names}

        # Calculate Chinese character ratio
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        chinese_char_ratio = chinese_chars / max(1, char_count)

        # (A) Text Structure Features
        avg_sentence_length = char_count / max(1, sentence_count)
        
        if language == 'zh':
            # For Chinese, word length is based on characters in the word
            avg_word_length = sum(len(w) for w in words) / word_count
        else:
            avg_word_length = sum(len(w) for w in words) / word_count
        
        sentence_lengths = [len(self.tokenize(s, language)) for s in sentences]
        sentence_length_variance = np.var(sentence_lengths) if sentence_lengths else 0
        avg_words_per_sentence = word_count / max(1, sentence_count)

        # (B) Vocabulary & Repetition Features
        unique_words = set(words)
        type_token_ratio = len(unique_words) / word_count
        
        # Hapax legomena: words that appear only once
        word_freq = Counter(words)
        hapax_count = sum(1 for count in word_freq.values() if count == 1)
        hapax_legomena_ratio = hapax_count / word_count
        
        # Function word ratio
        function_words = self.get_function_words(language)
        function_word_count = sum(1 for w in words if w.lower() in function_words or w in function_words)
        function_word_ratio = function_word_count / word_count
        
        # Lexical density: content words / total words
        content_word_count = word_count - function_word_count
        lexical_density = content_word_count / word_count

        # (C) Character-level Features
        if language == 'zh':
            punctuation_pattern = r'[，。！？；：、""''（）【】《》…—]'
        else:
            punctuation_pattern = r'[.,!?;:\'"()\-\[\]{}]'
        
        punctuation_count = len(re.findall(punctuation_pattern, text))
        punctuation_ratio = punctuation_count / max(1, char_count)
        
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / max(1, char_count)
        
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / max(1, char_count)

        # (D) Entropy Features
        char_entropy = self.calculate_entropy(chars)
        word_entropy = self.calculate_entropy(words)
        
        bigrams = self.get_ngrams(words, 2)
        bigram_entropy = self.calculate_entropy(bigrams)

        # Trigram repetition rate
        trigrams = self.get_ngrams(words, 3)
        if trigrams:
            trigram_counter = Counter(trigrams)
            repeated_trigrams = sum(1 for count in trigram_counter.values() if count > 1)
            trigram_repetition_rate = repeated_trigrams / len(trigram_counter)
        else:
            trigram_repetition_rate = 0.0

        # (E) Readability Features
        if language == 'en':
            total_syllables = sum(self.count_syllables(w) for w in words)
            avg_syllables_per_word = total_syllables / word_count
            long_word_count = sum(1 for w in words if len(w) > 6)
        else:
            # For Chinese, use character count as proxy
            avg_syllables_per_word = avg_word_length
            long_word_count = sum(1 for w in words if len(w) > 2)
        
        long_word_ratio = long_word_count / word_count
        
        # (F) Formal word ratio (AI indicator)
        formal_word_ratio = self.get_formal_indicators(text, language)

        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'sentence_count': sentence_count,
            'word_count': word_count,
            'char_count': char_count,
            'type_token_ratio': type_token_ratio,
            'hapax_legomena_ratio': hapax_legomena_ratio,
            'function_word_ratio': function_word_ratio,
            'punctuation_ratio': punctuation_ratio,
            'uppercase_ratio': uppercase_ratio,
            'digit_ratio': digit_ratio,
            'char_entropy': char_entropy,
            'word_entropy': word_entropy,
            'bigram_entropy': bigram_entropy,
            'trigram_repetition_rate': trigram_repetition_rate,
            'avg_words_per_sentence': avg_words_per_sentence,
            'sentence_length_variance': sentence_length_variance,
            'lexical_density': lexical_density,
            'avg_syllables_per_word': avg_syllables_per_word,
            'long_word_ratio': long_word_ratio,
            'chinese_char_ratio': chinese_char_ratio,
            'formal_word_ratio': formal_word_ratio
        }

    def extract_features_array(self, text: str, language: Optional[str] = None) -> np.ndarray:
        """Extract features as a numpy array for model input."""
        features = self.extract_features(text, language)
        return np.array([features[name] for name in self.feature_names])

    def extract_batch(self, texts: List[str], languages: Optional[List[str]] = None) -> np.ndarray:
        """Extract features for multiple texts."""
        if languages is None:
            return np.array([self.extract_features_array(text) for text in texts])
        return np.array([self.extract_features_array(text, lang) for text, lang in zip(texts, languages)])

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return descriptions of each feature for explainability."""
        return {
            'avg_sentence_length': 'Average number of characters per sentence / 平均每句字元數',
            'avg_word_length': 'Average number of characters per word / 平均每詞字元數',
            'sentence_count': 'Total number of sentences / 總句數',
            'word_count': 'Total number of words / 總詞數',
            'char_count': 'Total number of characters / 總字元數',
            'type_token_ratio': 'Vocabulary richness (unique words / total) / 詞彙豐富度',
            'hapax_legomena_ratio': 'Words appearing only once / total / 單次出現詞比例',
            'function_word_ratio': 'Proportion of function words / 虛詞比例',
            'punctuation_ratio': 'Proportion of punctuation / 標點符號比例',
            'uppercase_ratio': 'Proportion of uppercase characters / 大寫字母比例',
            'digit_ratio': 'Proportion of digit characters / 數字比例',
            'char_entropy': 'Shannon entropy of characters / 字元熵',
            'word_entropy': 'Shannon entropy of words / 詞彙熵',
            'bigram_entropy': 'Shannon entropy of word bigrams / 雙詞組熵',
            'trigram_repetition_rate': 'Rate of repeated word trigrams / 三詞組重複率',
            'avg_words_per_sentence': 'Average words per sentence / 平均每句詞數',
            'sentence_length_variance': 'Variance in sentence lengths / 句長變異度',
            'lexical_density': 'Content words / total words / 詞彙密度',
            'avg_syllables_per_word': 'Average syllables per word / 平均音節數',
            'long_word_ratio': 'Proportion of long words / 長詞比例',
            'chinese_char_ratio': 'Proportion of Chinese characters / 中文字元比例',
            'formal_word_ratio': 'Proportion of formal/academic words / 正式用語比例'
        }
