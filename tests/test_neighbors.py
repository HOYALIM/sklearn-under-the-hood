"""
Tests for neighbors implementations.
"""

import numpy as np
import pytest
from sklearn_from_scratch.neighbors import KNeighborsClassifier, KNeighborsRegressor


class TestKNeighborsClassifier:
    """Tests for KNeighborsClassifier."""
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality."""
        X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)
        
        # Test on training data
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy == 1.0  # Should perfectly classify training data
        
    def test_new_predictions(self):
        """Test predictions on new data."""
        X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)
        
        # New point close to class 0
        X_new = np.array([[1.5, 1.5]])
        pred = model.predict(X_new)
        assert pred[0] == 0
        
        # New point close to class 1
        X_new = np.array([[10.5, 10.5]])
        pred = model.predict(X_new)
        assert pred[0] == 1
        
    def test_manhattan_distance(self):
        """Test with Manhattan distance."""
        X = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
        model.fit(X, y)
        
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy == 1.0


class TestKNeighborsRegressor:
    """Tests for KNeighborsRegressor."""
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        model = KNeighborsRegressor(n_neighbors=2)
        model.fit(X, y)
        
        # Prediction at training point should be reasonably close
        pred = model.predict(np.array([[3]]))
        assert abs(pred[0] - 6) <= 1.5
        
    def test_averaging(self):
        """Test that predictions are averages of neighbors."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10, 20, 30, 40, 50])
        
        model = KNeighborsRegressor(n_neighbors=2)
        model.fit(X, y)
        
        # Point 2.5 should average neighbors at 2 and 3
        pred = model.predict(np.array([[2.5]]))
        expected = (20 + 30) / 2
        assert abs(pred[0] - expected) < 1.0
