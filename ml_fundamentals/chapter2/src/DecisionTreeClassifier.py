import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    """
    An implementation of a Decision Tree Classifier built from scratch.
    
    A Decision Tree is a non-parametric supervised learning method used for classification.
    The goal is to create a model that predicts the class of a target variable by learning 
    simple decision rules inferred from the data features.
    
    Mathematical Intuition:
    1. **Splitting Criterion**: The tree uses Information Gain based on Entropy to evaluate 
       the quality of a split. For a given node, it finds the feature and threshold that 
       maximizes the reduction in entropy:
       
       `Information Gain = Entropy(Parent) - [Weighted Average] * Entropy(Children)`
       
    2. **Entropy**: A measure of impurity or randomness in the data.
       `Entropy(S) = -Σ p_i * log2(p_i)` where p_i is the probability of class i.
       
    3. **Recursive Partitioning**: The dataset is recursively split into subsets until 
       a stopping criterion is met (e.g., maximum depth, minimum samples, or pure node).
    4. **Leaf Prediction**: The prediction for any instance reaching a leaf node is the 
       most common class label in that leaf.
       
    References to learn more:
    - "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
    - Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/tree.html#classification
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        """
        Initialize the Decision Tree Classifier.
        
        Args:
            max_depth: Maximum depth of the tree.
            min_samples_split: Minimum number of samples required to split an internal node.
            max_features: Number of features to consider when looking for the best split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).
        """
        self.n_features = X.shape[1]
        self.features = np.arange(self.n_features)
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursive method to build the tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            return self._most_common_label(y)

        # Feature bagging if max_features is specified
        feat_idxs = np.random.choice(self.features,
                                     self.max_features or n_features,
                                     replace=False)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            return self._most_common_label(y)

        # Split the data
        left_idxs = X[:, best_feature] <= best_thresh
        right_idxs = X[:, best_feature] > best_thresh

        # Recursive calls
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return (best_feature, best_thresh, left, right)
    
    def _most_common_label(self, y):
        """
        Return the most frequent label in y.
        """
        if len(y) == 0:
            return None
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _entropy(self, y):
        """
        Calculate entropy of a label array.
        """
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _information_gain(self, y, left_y, right_y):
        """
        Calculate Information Gain.
        """
        parent_entropy = self._entropy(y)
        n = len(y)
        n_l, n_r = len(left_y), len(right_y)
        
        if n_l == 0 or n_r == 0:
            return 0
        
        child_entropy = (n_l / n) * self._entropy(left_y) + (n_r / n) * self._entropy(right_y)
        return parent_entropy - child_entropy

    def _best_split(self, X, y, feat_idxs):
        """
        Find the best feature and threshold for splitting.
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                # Potential split
                left_y = y[X[:, feat] <= t]
                right_y = y[X[:, feat] > t]

                gain = self._information_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = t

        return split_idx, split_thresh

    def predict(self, X):
        """
        Predict classes for X.
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.
        """
        if not isinstance(node, tuple):
            return node

        feature, thresh, left, right = node

        if x[feature] <= thresh:
            return self._traverse_tree(x, left)
        return self._traverse_tree(x, right)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load sample data
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit
    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {acc:.4f}")