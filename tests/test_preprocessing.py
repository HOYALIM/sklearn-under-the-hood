"""
Tests for preprocessing implementations.
"""

import numpy as np
import pytest
from sklearn_from_scratch.preprocessing import (
    StandardScaler, MinMaxScaler, train_test_split
)


class TestStandardScaler:
    """Tests for StandardScaler."""
    
    def test_fit_transform(self):
        """Test basic fit and transform."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # After standardization, mean should be ~0 and std should be ~1
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)
        
    def test_transform_separate(self):
        """Test separate fit and transform."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[7, 8]])
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Mean of training data should be ~0
        assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10)
        
        # Test data should use same scaling parameters
        assert X_test_scaled.shape == X_test.shape


class TestMinMaxScaler:
    """Tests for MinMaxScaler."""
    
    def test_fit_transform(self):
        """Test basic fit and transform."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # After min-max scaling, all values should be in [0, 1]
        assert np.all(X_scaled >= 0)
        assert np.all(X_scaled <= 1)
        
        # Min values should be 0, max values should be 1
        assert np.allclose(np.min(X_scaled, axis=0), 0)
        assert np.allclose(np.max(X_scaled, axis=0), 1)
        
    def test_custom_range(self):
        """Test with custom feature range."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        
        # All values should be in [-1, 1]
        assert np.all(X_scaled >= -1)
        assert np.all(X_scaled <= 1)
        
        # Min values should be -1, max values should be 1
        assert np.allclose(np.min(X_scaled, axis=0), -1)
        assert np.allclose(np.max(X_scaled, axis=0), 1)


class TestTrainTestSplit:
    """Tests for train_test_split."""
    
    def test_split_sizes(self):
        """Test that split produces correct sizes."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
    def test_random_state(self):
        """Test that random_state produces reproducible results."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Same random_state should give same split
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
        
    def test_no_overlap(self):
        """Test that train and test sets don't overlap."""
        X = np.arange(20).reshape(20, 1)
        y = np.arange(20)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Check no overlap
        train_set = set(y_train)
        test_set = set(y_test)
        assert len(train_set.intersection(test_set)) == 0
