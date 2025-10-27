"""
Tests for linear models implementations.
"""

import numpy as np
import pytest
from sklearn_from_scratch.linear_models import LinearRegression, LogisticRegression


class TestLinearRegression:
    """Tests for LinearRegression."""
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality."""
        # Simple linear data: y = 2x + 1
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([3, 5, 7, 9, 11])
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Check predictions
        predictions = model.predict(X)
        np.testing.assert_array_almost_equal(predictions, y, decimal=5)
        
    def test_coefficients(self):
        """Test that coefficients are learned correctly."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([3, 5, 7, 9, 11])
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Should learn slope ≈ 2 and intercept ≈ 1
        assert abs(model.coef_[0] - 2.0) < 0.1
        assert abs(model.intercept_ - 1.0) < 0.1
        
    def test_score(self):
        """Test R² score calculation."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([3, 5, 7, 9, 11])
        
        model = LinearRegression()
        model.fit(X, y)
        
        score = model.score(X, y)
        assert score > 0.99  # Should be very close to 1 for perfect fit
        
    def test_no_intercept(self):
        """Test LinearRegression without intercept."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        assert abs(model.coef_[0] - 2.0) < 0.1
        assert model.intercept_ == 0.0


class TestLogisticRegression:
    """Tests for LogisticRegression."""
    
    def test_fit_predict(self):
        """Test basic fit and predict functionality."""
        # Simple binary classification
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, max_iter=1000)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Should predict at least 4 out of 6 correctly
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.6
        
    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, max_iter=1000)
        model.fit(X, y)
        
        probabilities = model.predict_proba(X)
        
        # Probabilities should be between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        
        # Lower X values should have lower probabilities
        assert probabilities[0] < probabilities[-1]
        
    def test_score(self):
        """Test accuracy score calculation."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, max_iter=1000)
        model.fit(X, y)
        
        score = model.score(X, y)
        assert score > 0.6  # Should get decent accuracy
