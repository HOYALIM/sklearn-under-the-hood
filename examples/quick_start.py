"""
Quick Start Example - Getting started with sklearn from scratch

This script demonstrates basic usage of the custom ML implementations.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression

# Import custom implementations
from sklearn_from_scratch.linear_models import LinearRegression, LogisticRegression
from sklearn_from_scratch.neighbors import KNeighborsClassifier
from sklearn_from_scratch.tree import DecisionTreeClassifier
from sklearn_from_scratch.preprocessing import StandardScaler, train_test_split
from sklearn_from_scratch.metrics import accuracy_score, r2_score


def demo_linear_regression():
    """Demonstrate Linear Regression."""
    print("\n" + "="*60)
    print("LINEAR REGRESSION EXAMPLE")
    print("="*60)
    
    # Create simple data: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    print(f"\nData: y = 2x + 1")
    print(f"Learned coefficient: {model.coef_[0]:.4f}")
    print(f"Learned intercept: {model.intercept_:.4f}")
    print(f"R² Score: {model.score(X, y):.6f}")
    print(f"Predictions: {predictions}")


def demo_logistic_regression():
    """Demonstrate Logistic Regression."""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION EXAMPLE")
    print("="*60)
    
    # Generate classification data
    X, y = make_classification(n_samples=200, n_features=4, 
                              n_informative=2, n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Show probability predictions
    X_sample = X_test[:5]
    y_sample = y_test[:5]
    probs = model.predict_proba(X_sample)
    preds = model.predict(X_sample)
    
    print(f"\nSample Predictions:")
    for i in range(len(X_sample)):
        print(f"  Sample {i+1}: True={y_sample[i]}, Pred={preds[i]}, Prob={probs[i]:.4f}")


def demo_knn():
    """Demonstrate K-Nearest Neighbors."""
    print("\n" + "="*60)
    print("K-NEAREST NEIGHBORS EXAMPLE")
    print("="*60)
    
    # Generate data with clear clusters
    X, y = make_classification(n_samples=100, n_features=2, 
                              n_informative=2, n_redundant=0,
                              n_clusters_per_class=1, class_sep=2.0,
                              random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Compare different K values
    print("\nComparing different K values:")
    for k in [1, 3, 5, 10]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"  K={k:2d}: Accuracy = {accuracy:.4f}")


def demo_decision_tree():
    """Demonstrate Decision Tree."""
    print("\n" + "="*60)
    print("DECISION TREE EXAMPLE")
    print("="*60)
    
    # Generate classification data
    X, y = make_classification(n_samples=200, n_features=4,
                              n_informative=2, n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Compare different depths
    print("\nComparing different tree depths:")
    for depth in [2, 5, 10, None]:
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        depth_str = 'unlimited' if depth is None else f'{depth:2d}'
        print(f"  Depth={depth_str}: Train={train_acc:.4f}, Test={test_acc:.4f}")


def demo_preprocessing():
    """Demonstrate data preprocessing."""
    print("\n" + "="*60)
    print("DATA PREPROCESSING EXAMPLE")
    print("="*60)
    
    # Generate data with different scales
    np.random.seed(42)
    X = np.array([
        [100, 0.01],
        [200, 0.02],
        [300, 0.03],
        [400, 0.04],
        [500, 0.05]
    ])
    
    print("\nOriginal data:")
    print(f"  Mean: {X.mean(axis=0)}")
    print(f"  Std:  {X.std(axis=0)}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nStandardized data:")
    print(f"  Mean: {X_scaled.mean(axis=0)}")
    print(f"  Std:  {X_scaled.std(axis=0)}")
    print("\n  ✅ Mean ≈ 0, Std ≈ 1 for all features!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SKLEARN FROM SCRATCH - QUICK START")
    print("="*60)
    print("\nDemonstrating custom ML implementations")
    
    demo_linear_regression()
    demo_logistic_regression()
    demo_knn()
    demo_decision_tree()
    demo_preprocessing()
    
    print("\n" + "="*60)
    print("QUICK START COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check out notebooks/algorithms_demo.ipynb for visualizations")
    print("  2. Read tutorials/ML_ALGORITHMS_GUIDE.md for theory")
    print("  3. Run examples/compare_with_sklearn.py for comparisons")
    print("  4. Explore the source code in sklearn_from_scratch/")
    print()
