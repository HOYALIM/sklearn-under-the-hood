"""
Comparison example: Custom implementation vs scikit-learn

This script demonstrates the comparison between our from-scratch implementations
and the official scikit-learn implementations.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split as sklearn_split
import sklearn.linear_model as sklearn_linear
import sklearn.neighbors as sklearn_neighbors
import sklearn.tree as sklearn_tree
import time

from sklearn_from_scratch.linear_models import LinearRegression, LogisticRegression
from sklearn_from_scratch.neighbors import KNeighborsClassifier
from sklearn_from_scratch.tree import DecisionTreeClassifier
from sklearn_from_scratch.preprocessing import train_test_split


def compare_linear_regression():
    """Compare Linear Regression implementations."""
    print("\n" + "="*70)
    print("LINEAR REGRESSION COMPARISON")
    print("="*70)
    
    # Generate regression data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Our implementation
    print("\n📊 Custom Implementation:")
    start = time.time()
    custom_model = LinearRegression()
    custom_model.fit(X_train, y_train)
    custom_score = custom_model.score(X_test, y_test)
    custom_time = time.time() - start
    print(f"   R² Score: {custom_score:.6f}")
    print(f"   Training Time: {custom_time:.6f}s")
    print(f"   Coefficients (first 3): {custom_model.coef_[:3]}")
    
    # Scikit-learn implementation
    print("\n🔬 Scikit-learn Implementation:")
    start = time.time()
    sklearn_model = sklearn_linear.LinearRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_score = sklearn_model.score(X_test, y_test)
    sklearn_time = time.time() - start
    print(f"   R² Score: {sklearn_score:.6f}")
    print(f"   Training Time: {sklearn_time:.6f}s")
    print(f"   Coefficients (first 3): {sklearn_model.coef_[:3]}")
    
    print(f"\n✅ Score Difference: {abs(custom_score - sklearn_score):.8f}")


def compare_logistic_regression():
    """Compare Logistic Regression implementations."""
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION COMPARISON")
    print("="*70)
    
    # Generate classification data
    X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                              n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Our implementation
    print("\n📊 Custom Implementation:")
    start = time.time()
    custom_model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    custom_model.fit(X_train, y_train)
    custom_score = custom_model.score(X_test, y_test)
    custom_time = time.time() - start
    print(f"   Accuracy: {custom_score:.6f}")
    print(f"   Training Time: {custom_time:.6f}s")
    
    # Scikit-learn implementation
    print("\n🔬 Scikit-learn Implementation:")
    start = time.time()
    sklearn_model = sklearn_linear.LogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train)
    sklearn_score = sklearn_model.score(X_test, y_test)
    sklearn_time = time.time() - start
    print(f"   Accuracy: {sklearn_score:.6f}")
    print(f"   Training Time: {sklearn_time:.6f}s")
    
    print(f"\n✅ Accuracy Difference: {abs(custom_score - sklearn_score):.6f}")


def compare_knn():
    """Compare K-Nearest Neighbors implementations."""
    print("\n" + "="*70)
    print("K-NEAREST NEIGHBORS COMPARISON")
    print("="*70)
    
    # Generate classification data
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                              n_redundant=1, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Our implementation
    print("\n📊 Custom Implementation:")
    start = time.time()
    custom_model = KNeighborsClassifier(n_neighbors=5)
    custom_model.fit(X_train, y_train)
    custom_score = custom_model.score(X_test, y_test)
    custom_time = time.time() - start
    print(f"   Accuracy: {custom_score:.6f}")
    print(f"   Training Time: {custom_time:.6f}s")
    
    # Scikit-learn implementation
    print("\n🔬 Scikit-learn Implementation:")
    start = time.time()
    sklearn_model = sklearn_neighbors.KNeighborsClassifier(n_neighbors=5)
    sklearn_model.fit(X_train, y_train)
    sklearn_score = sklearn_model.score(X_test, y_test)
    sklearn_time = time.time() - start
    print(f"   Accuracy: {sklearn_score:.6f}")
    print(f"   Training Time: {sklearn_time:.6f}s")
    
    print(f"\n✅ Accuracy Difference: {abs(custom_score - sklearn_score):.6f}")


def compare_decision_tree():
    """Compare Decision Tree implementations."""
    print("\n" + "="*70)
    print("DECISION TREE COMPARISON")
    print("="*70)
    
    # Generate classification data
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                              n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Our implementation
    print("\n📊 Custom Implementation:")
    start = time.time()
    custom_model = DecisionTreeClassifier(max_depth=5, criterion='gini')
    custom_model.fit(X_train, y_train)
    custom_score = custom_model.score(X_test, y_test)
    custom_time = time.time() - start
    print(f"   Accuracy: {custom_score:.6f}")
    print(f"   Training Time: {custom_time:.6f}s")
    
    # Scikit-learn implementation
    print("\n🔬 Scikit-learn Implementation:")
    start = time.time()
    sklearn_model = sklearn_tree.DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
    sklearn_model.fit(X_train, y_train)
    sklearn_score = sklearn_model.score(X_test, y_test)
    sklearn_time = time.time() - start
    print(f"   Accuracy: {sklearn_score:.6f}")
    print(f"   Training Time: {sklearn_time:.6f}s")
    
    print(f"\n✅ Accuracy Difference: {abs(custom_score - sklearn_score):.6f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SKLEARN FROM SCRATCH vs OFFICIAL SCIKIT-LEARN")
    print("Comparing custom implementations with official sklearn")
    print("="*70)
    
    compare_linear_regression()
    compare_logistic_regression()
    compare_knn()
    compare_decision_tree()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print("\nNotes:")
    print("- Custom implementations may be slower but achieve similar accuracy")
    print("- Scikit-learn uses optimized C/Cython code for better performance")
    print("- Both implementations follow the same mathematical principles")
    print()
