# Decision Tree Regression

A Decision Tree is a non-parametric supervised learning method used for both classification and regression. The `DecisionTreeRegressor` builds a model in the form of a tree structure by breaking down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.

## Mathematical Intuition

### 1. Splitting Criterion (Variance Reduction)
When choosing how to split a node, the algorithm looks for the feature and the threshold that result in the largest decrease in variance (equivalent to minimizing the Mean Squared Error). 
For a given node with data $D$, it evaluates the MSE of a split as the weighted sum of variances in the left and right child nodes:
$$ \text{MSE}_{split} = \frac{N_{left}}{N} \text{Var}(y_{left}) + \frac{N_{right}}{N} \text{Var}(y_{right}) $$
The split that minimizes this $\text{MSE}_{split}$ is chosen.

### 2. Recursive Partitioning
The dataset is recursively partitioned based on the best splits. The process continues until a stopping criterion is met, such as:
- Reaching a maximum depth (`max_depth`).
- Having fewer samples than the minimum required to split (`min_samples_split`).

### 3. Leaf Prediction
Once a leaf node is reached, the prediction for any new instance falling into that leaf is the mean value of the target variables of the training instances in that leaf:
$$ \hat{y} = \frac{1}{N_{leaf}} \sum_{i \in \text{leaf}} y_i $$

---

## How to Use

The `DecisionTreeRegressor` class in `src/DecisionTreeRegression.py` provides a standalone implementation that does not rely on configuration files, making it highly modular.

### Implementation Example
```python
import numpy as np
from ml_fundamentals.chapter1.src.DecisionTreeRegression import DecisionTreeRegressor

# 1. Prepare Data
# X should be a 2D numpy array, y should be a 1D numpy array
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([2, 3, 4, 5])
X_test = np.array([[2.5, 3.5]])

# 2. Instantiate Model
# You can specify max_depth, min_samples_split, and max_features
tree = DecisionTreeRegressor(max_depth=5, min_samples_split=2)

# 3. Train Model
tree.fit(X_train, y_train)

# 4. Predict
predictions = tree.predict(X_test)
print("Predictions:", predictions)

---
**File Location:** `ml_fundamentals/chapter1/src/DecisionTreeRegression.py`
```
