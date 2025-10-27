"""
Tests for tree implementations.
"""

import numpy as np
import pytest
from sklearn_from_scratch.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier:
    """Tests for DecisionTreeClassifier."""
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality."""
        X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X, y)
        
        # Should perfectly classify training data
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy == 1.0
        
    def test_max_depth(self):
        """Test max_depth parameter."""
        X = np.array([[i] for i in range(100)])
        y = np.array([i % 2 for i in range(100)])
        
        # Shallow tree
        model_shallow = DecisionTreeClassifier(max_depth=1)
        model_shallow.fit(X, y)
        
        # Deep tree
        model_deep = DecisionTreeClassifier(max_depth=10)
        model_deep.fit(X, y)
        
        # Deep tree should fit better
        score_shallow = model_shallow.score(X, y)
        score_deep = model_deep.score(X, y)
        assert score_deep >= score_shallow
        
    def test_entropy_criterion(self):
        """Test with entropy criterion."""
        X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
        model.fit(X, y)
        
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy == 1.0
        
    def test_min_samples_split(self):
        """Test min_samples_split parameter."""
        X = np.array([[i] for i in range(10)])
        y = np.array([i % 2 for i in range(10)])
        
        model = DecisionTreeClassifier(min_samples_split=5)
        model.fit(X, y)
        
        # Should still make reasonable predictions
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.5
