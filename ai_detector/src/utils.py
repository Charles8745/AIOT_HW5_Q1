"""
Utility Functions for AI/Human Text Detection

This module contains helper functions for data loading,
preprocessing, visualization, and other common operations.
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


def load_dataset(file_path: str) -> Tuple[List[str], List[int]]:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file with 'text' and 'label' columns
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '').strip()
            label = int(row.get('label', 0))
            if text:
                texts.append(text)
                labels.append(label)
    
    return texts, labels


def load_dataset_pandas(file_path: str) -> pd.DataFrame:
    """
    Load dataset as pandas DataFrame.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with text and label columns
    """
    return pd.read_csv(file_path)


def save_predictions(predictions: List[Dict], output_path: str):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save CSV file
    """
    if not predictions:
        return
        
    keys = predictions[0].keys()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(predictions)


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text.strip()


def split_dataset(texts: List[str], labels: List[int], 
                  test_size: float = 0.2, random_state: int = 42) -> Dict:
    """
    Split dataset into train and test sets.
    
    Args:
        texts: List of texts
        labels: List of labels
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Dictionary with train/test splits
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def get_dataset_stats(texts: List[str], labels: List[int]) -> Dict:
    """
    Calculate dataset statistics.
    
    Args:
        texts: List of texts
        labels: List of labels
        
    Returns:
        Dictionary of statistics
    """
    human_count = labels.count(0)
    ai_count = labels.count(1)
    total = len(labels)
    
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]
    
    return {
        'total_samples': total,
        'human_count': human_count,
        'ai_count': ai_count,
        'human_ratio': human_count / total if total > 0 else 0,
        'ai_ratio': ai_count / total if total > 0 else 0,
        'avg_word_count': np.mean(word_counts),
        'std_word_count': np.std(word_counts),
        'min_word_count': np.min(word_counts),
        'max_word_count': np.max(word_counts),
        'avg_char_count': np.mean(char_counts),
        'std_char_count': np.std(char_counts)
    }


def format_prediction_result(result: Dict) -> str:
    """
    Format prediction result for display.
    
    Args:
        result: Prediction dictionary
        
    Returns:
        Formatted string
    """
    lines = [
        f"Prediction: {result.get('label_str', 'Unknown')}",
        f"AI Probability: {result.get('ai_probability', 0):.2%}",
        f"Human Probability: {result.get('human_probability', 0):.2%}",
        f"Confidence: {result.get('confidence', 0):.2%}"
    ]
    return '\n'.join(lines)


def format_ensemble_result(result: Dict) -> str:
    """
    Format ensemble prediction result for display.
    
    Args:
        result: Ensemble prediction dictionary
        
    Returns:
        Formatted string
    """
    lines = ["Model Predictions:"]
    
    for model, prob in result.get('model_probabilities', {}).items():
        lines.append(f"  {model}: {prob:.2%}")
    
    lines.extend([
        "",
        f"Ensemble Probability: {result.get('ensemble_probability', 0):.2%}",
        f"Final Decision: {result.get('final_label_str', 'Unknown')}",
        f"Confidence: {result.get('confidence', 0):.2%}"
    ])
    
    return '\n'.join(lines)


def create_sample_dataset(output_path: str, n_samples: int = 20):
    """
    Create a sample dataset for testing.
    
    Args:
        output_path: Path to save the dataset
        n_samples: Number of samples to create
    """
    # Sample human-written texts (more casual, varied)
    human_texts = [
        "I went to the store yesterday and bought some groceries. The weather was nice so I walked instead of driving. It felt good to be outside after being stuck indoors all week.",
        "My dog absolutely loves going to the park! She gets so excited when she sees other dogs to play with. Sometimes she's a bit too energetic and knocks things over, but I can't stay mad at her.",
        "Just finished reading this amazing book about space exploration. I couldn't put it down! The author really knows how to tell a story. Would definitely recommend it to anyone who likes science fiction.",
        "Cooking dinner tonight was a disaster. I burned the rice and the vegetables were way too salty. Oh well, at least the family was nice about it. We ended up ordering pizza instead lol",
        "The concert last night was incredible!! The band played all my favorite songs and the crowd energy was amazing. My voice is completely gone from all the singing along haha",
        "Working from home has its pros and cons. I love not having to commute but sometimes I really miss chatting with my coworkers. The coffee at the office was way better too.",
        "My kids are driving me crazy today. Summer break just started and they're already bored. Trying to think of activities that don't involve screens is harder than I thought it would be.",
        "Finally got around to cleaning out the garage this weekend. Found so much stuff I forgot we had! It's kind of like a trip down memory lane, honestly.",
        "The new coffee shop downtown is pretty good. The barista was super friendly and they have this cool vintage decor. Prices are a bit steep though.",
        "Can't believe it's already June. This year is flying by so fast! Feels like just yesterday we were celebrating New Year's. Time is weird like that I guess."
    ]
    
    # Sample AI-generated texts (more formal, structured)
    ai_texts = [
        "Artificial intelligence represents a transformative technology that continues to reshape various aspects of modern society. The integration of AI systems into everyday applications has demonstrated significant potential for improving efficiency and decision-making processes. Furthermore, ongoing research in machine learning promises even greater advancements.",
        "Climate change poses one of the most significant challenges facing humanity in the 21st century. Scientific consensus indicates that human activities have contributed substantially to global warming. Addressing this issue requires coordinated international efforts and significant changes to our energy infrastructure and consumption patterns.",
        "The digital revolution has fundamentally altered how we communicate, work, and access information. Social media platforms have created new paradigms for human interaction and information sharing. These changes present both opportunities and challenges for individuals and organizations seeking to adapt to an increasingly connected world.",
        "Education systems worldwide are undergoing significant transformation in response to technological advancement. The integration of digital tools and online learning platforms has expanded access to educational resources. Additionally, there is growing recognition of the importance of developing critical thinking and adaptability skills in students.",
        "Healthcare innovation continues to advance at a remarkable pace, with new treatments and technologies offering hope for previously untreatable conditions. Precision medicine, enabled by genetic sequencing and data analytics, represents a paradigm shift in disease prevention and treatment approaches.",
        "Economic globalization has created unprecedented interconnectedness between markets and nations. International trade agreements and technological advances have facilitated the free flow of goods, services, and capital across borders. This integration has generated both economic opportunities and challenges for different regions.",
        "Sustainable development requires balancing economic growth with environmental protection and social equity. Organizations and governments are increasingly recognizing the importance of implementing sustainable practices. This transition presents significant opportunities for innovation and long-term value creation.",
        "The evolution of transportation technology is driving fundamental changes in urban mobility patterns. Electric vehicles, autonomous driving systems, and shared transportation services are reshaping how people move through cities. These developments have important implications for urban planning and infrastructure investment.",
        "Digital security has become a critical concern for individuals, organizations, and governments alike. The increasing sophistication of cyber threats requires continuous advancement in protective measures and security protocols. Effective cybersecurity strategies must address both technical and human factors.",
        "The future of work is being shaped by automation, artificial intelligence, and changing workforce demographics. Organizations must adapt their structures and practices to remain competitive in an evolving landscape. Developing a skilled and adaptable workforce is essential for long-term success."
    ]
    
    # Combine and create dataset
    data = []
    
    for text in human_texts:
        data.append({'text': text, 'label': 0})
    
    for text in ai_texts:
        data.append({'text': text, 'label': 1})
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Created sample dataset with {len(data)} samples at {output_path}")


def get_label_name(label: int) -> str:
    """Convert numeric label to string."""
    return 'AI-generated' if label == 1 else 'Human-written'


def get_color_for_probability(prob: float) -> str:
    """
    Get color for probability value (for visualization).
    
    Args:
        prob: Probability value (0-1)
        
    Returns:
        Color string
    """
    if prob >= 0.7:
        return '#ff4444'  # Red for high AI probability
    elif prob >= 0.5:
        return '#ffaa00'  # Orange for medium-high
    elif prob >= 0.3:
        return '#ffff00'  # Yellow for medium
    else:
        return '#44ff44'  # Green for low (human)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path() -> Path:
    """Get the data directory path."""
    return get_project_root() / 'data'


def get_models_path() -> Path:
    """Get the models directory path."""
    path = get_project_root() / 'models'
    path.mkdir(exist_ok=True)
    return path
