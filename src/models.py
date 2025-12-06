"""
Machine Learning Models for AI/Human Text Detection

This module contains TF-IDF + Logistic Regression and
Random Forest classifiers with feature importance.
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


class TFIDFDetector:
    """
    TF-IDF based AI text detector using Logistic Regression.
    Provides probability scores and feature weights for explainability.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the TF-IDF detector.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.is_trained = False
        
    def train(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Train the TF-IDF detector.
        
        Args:
            texts: List of text samples
            labels: List of labels (0=Human, 1=AI)
            
        Returns:
            Dictionary of training metrics
        """
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        # Calculate training metrics
        predictions = self.classifier.predict(X)
        probs = self.classifier.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
            'roc_auc': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
        }
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict if text is AI-generated.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        X = self.vectorizer.transform([text])
        prob = self.classifier.predict_proba(X)[0]
        label = self.classifier.predict(X)[0]
        
        return {
            'label': int(label),
            'label_str': 'AI-generated' if label == 1 else 'Human-written',
            'ai_probability': float(prob[1]),
            'human_probability': float(prob[0]),
            'confidence': float(max(prob))
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict for multiple texts."""
        return [self.predict(text) for text in texts]
    
    def get_top_features(self, n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features indicating AI or Human text.
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary with 'ai_indicators' and 'human_indicators'
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.classifier.coef_[0]
        
        # Sort by coefficient value
        sorted_indices = np.argsort(coefficients)
        
        # Top AI indicators (positive coefficients)
        ai_indices = sorted_indices[-n:][::-1]
        ai_indicators = [(feature_names[i], float(coefficients[i])) for i in ai_indices]
        
        # Top Human indicators (negative coefficients)
        human_indices = sorted_indices[:n]
        human_indicators = [(feature_names[i], float(coefficients[i])) for i in human_indices]
        
        return {
            'ai_indicators': ai_indicators,
            'human_indicators': human_indicators
        }
    
    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.is_trained = data['is_trained']


class RandomForestDetector:
    """
    Random Forest classifier using hand-crafted features.
    Provides feature importance for explainability.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 10):
        """
        Initialize the Random Forest detector.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        """
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.feature_names: List[str] = []
        self.is_trained = False
        
    def train(self, X: np.ndarray, labels: List[int], 
              feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Train the Random Forest detector.
        
        Args:
            X: Feature matrix
            labels: List of labels (0=Human, 1=AI)
            feature_names: Names of features
            
        Returns:
            Dictionary of training metrics
        """
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Calculate training metrics
        predictions = self.classifier.predict(X)
        probs = self.classifier.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
            'roc_auc': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
        }
    
    def predict(self, X: np.ndarray) -> Dict[str, float]:
        """
        Predict if features indicate AI-generated text.
        
        Args:
            X: Feature array (1D or 2D)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        prob = self.classifier.predict_proba(X)[0]
        label = self.classifier.predict(X)[0]
        
        return {
            'label': int(label),
            'label_str': 'AI-generated' if label == 1 else 'Human-written',
            'ai_probability': float(prob[1]),
            'human_probability': float(prob[0]),
            'confidence': float(max(prob))
        }
    
    def predict_batch(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Predict for multiple feature arrays."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        results = []
        probs = self.classifier.predict_proba(X)
        labels = self.classifier.predict(X)
        
        for i in range(len(labels)):
            results.append({
                'label': int(labels[i]),
                'label_str': 'AI-generated' if labels[i] == 1 else 'Human-written',
                'ai_probability': float(probs[i, 1]),
                'human_probability': float(probs[i, 0]),
                'confidence': float(max(probs[i]))
            })
        
        return results
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        importances = self.classifier.feature_importances_
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance
    
    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.feature_names = data['feature_names']
            self.is_trained = data['is_trained']


class EnsembleDetector:
    """
    Ensemble detector combining multiple models using soft voting.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble detector.
        
        Args:
            weights: Dictionary of model weights for voting
        """
        self.weights = weights or {
            'tfidf': 0.33,
            'rf': 0.33,
            'transformer': 0.34
        }
        
    def combine_predictions(self, predictions: Dict[str, Dict]) -> Dict:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: Dictionary with model names as keys and prediction dicts as values
            
        Returns:
            Combined ensemble prediction
        """
        total_weight = 0
        weighted_ai_prob = 0
        
        model_results = {}
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1.0 / len(predictions))
            ai_prob = pred.get('ai_probability', 0.5)
            
            weighted_ai_prob += weight * ai_prob
            total_weight += weight
            
            model_results[model_name] = ai_prob
        
        if total_weight > 0:
            ensemble_prob = weighted_ai_prob / total_weight
        else:
            ensemble_prob = 0.5
            
        final_label = 1 if ensemble_prob >= 0.5 else 0
        
        return {
            'model_probabilities': model_results,
            'ensemble_probability': ensemble_prob,
            'final_label': final_label,
            'final_label_str': 'AI-generated' if final_label == 1 else 'Human-written',
            'confidence': abs(ensemble_prob - 0.5) * 2  # Scale to 0-1
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """Update model weights."""
        self.weights = weights


def evaluate_model(y_true: List[int], y_pred: List[int], 
                   y_prob: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None and len(set(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Get confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_roc_curve(y_true: List[int], y_prob: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get ROC curve data."""
    return roc_curve(y_true, y_prob)
