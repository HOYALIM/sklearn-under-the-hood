"""
Tests for metrics implementations.
"""

import numpy as np
import pytest
from sklearn_from_scratch.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, precision_score, recall_score, f1_score
)


class TestClassificationMetrics:
    """Tests for classification metrics."""
    
    def test_accuracy_score(self):
        """Test accuracy score calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 0.8  # 4 out of 5 correct
        
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Expected: [[1, 1], [1, 1]]
        assert cm[0, 0] == 1  # True negatives
        assert cm[0, 1] == 1  # False positives
        assert cm[1, 0] == 1  # False negatives
        assert cm[1, 1] == 1  # True positives
        
    def test_precision_score(self):
        """Test precision score calculation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        precision = precision_score(y_true, y_pred)
        # TP=2, FP=1 -> precision = 2/3
        assert abs(precision - 2/3) < 0.01
        
    def test_recall_score(self):
        """Test recall score calculation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        recall = recall_score(y_true, y_pred)
        # TP=2, FN=1 -> recall = 2/3
        assert abs(recall - 2/3) < 0.01
        
    def test_f1_score(self):
        """Test F1 score calculation."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])
        
        f1 = f1_score(y_true, y_pred)
        assert f1 == 1.0  # Perfect classification


class TestRegressionMetrics:
    """Tests for regression metrics."""
    
    def test_mean_squared_error(self):
        """Test MSE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        mse = mean_squared_error(y_true, y_pred)
        assert mse == 0.0  # Perfect predictions
        
        y_pred = np.array([2, 3, 4, 5, 6])
        mse = mean_squared_error(y_true, y_pred)
        assert mse == 1.0  # Off by 1 each
        
    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 1.0
        
    def test_r2_score(self):
        """Test R² score calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = r2_score(y_true, y_pred)
        assert r2 == 1.0  # Perfect predictions
        
        y_pred = np.array([3, 3, 3, 3, 3])  # Always predict mean
        r2 = r2_score(y_true, y_pred)
        assert r2 == 0.0  # No better than mean
