# Decision Tree Classifier

A Decision Tree is a non-parametric supervised learning method that predicts the class of a target variable by learning simple decision rules inferred from the data features. It recursively partitions the feature space into pure subsets.

## Mathematical Intuition

### 1. Splitting Criterion (Information Gain)
The tree needs to decide which feature and threshold best splits the data at each node. It evaluates the quality of a split using **Information Gain**, which measures the reduction in entropy (uncertainty) after the split.
$$ \text{Information Gain} = \text{Entropy}(Parent) - \sum_{j} \frac{N_j}{N} \text{Entropy}(Child_j) $$

### 2. Entropy
Entropy is a measure of impurity or randomness in the data. A perfectly pure node (all samples belong to one class) has an entropy of 0.
$$ \text{Entropy}(S) = -\sum_{i} p_i \log_2(p_i) $$
where $p_i$ is the proportion of samples belonging to class $i$ in node $S$.

### 3. Recursive Partitioning & Prediction
The dataset is recursively split into subsets until a stopping criterion is met (e.g., maximum tree depth or minimum samples to split). For prediction, an instance traverses the tree based on the learned rules, and the prediction is the most common class label in the leaf node it reaches.

---

## How to Use

The `DecisionTreeClassifier` class in `src/DecisionTreeClassifier.py` implements the core tree logic. (Note: It is used natively by the Random Forest implementation and doesn't wrap the configuration logic itself).

### Implementation Example
```python
from ml_fundamentals.chapter2.src.DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np

# Example Data
X_train = np.array([[2, 3], [10, 15], [3, 2], [8, 11]])
y_train = np.array([0, 1, 0, 1])

X_test = np.array([[1, 2], [9, 12]])

# 1. Instantiate Model
clf = DecisionTreeClassifier(max_depth=5)

# 2. Train Model
clf.fit(X_train, y_train)

# 3. Predict
y_pred = clf.predict(X_test)
print(y_pred) # Output: [0, 1]
```

---
**File Location:** `ml_fundamentals/chapter2/src/DecisionTreeClassifier.py`
