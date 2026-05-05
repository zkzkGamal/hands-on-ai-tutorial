# Random Forest Regression

A Random Forest is a powerful ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the average prediction of the individual trees for regression tasks. By leveraging multiple uncorrelated trees, it significantly reduces the variance and risk of overfitting compared to a single decision tree.

## Mathematical Intuition

### 1. Bootstrapping
For each tree in the forest, a random subset of the training data is sampled *with replacement*. This technique is called bootstrapping. 
If the original dataset has size $N$, each tree is trained on a bootstrap sample also of size $N$, but some original samples may appear multiple times while others may not appear at all.

### 2. Random Subspaces
When splitting a node during the construction of any given tree, the algorithm does not search for the best feature among all available features. Instead, only a random subset of features is considered for splitting at each node. 
This process decorrelates the trees, ensuring that the forest is composed of diverse decision paths.

### 3. Aggregation
Once all $M$ trees are built, the final prediction for a new instance $x$ is calculated as the unweighted average of the predictions from all the individual trees:
$$ \hat{y} = \frac{1}{M} \sum_{m=1}^{M} f_m(x) $$
where $f_m(x)$ is the prediction of the $m$-th decision tree.

---

## How to Use

The `RandomForestRegression` class in `src/RandomForestRegression.py` provides a full pipeline relying on an underlying configuration file and the custom `DecisionTreeRegressor` class.

### 1. Configuration Setup
The class uses the `Random Forest Regression` section under `models` to retrieve parameters like `n_estimators` (number of trees), `max_depth`, and `max_features`:
```json
{
  "datasets": [{"path": "data.csv", "target": "target_column"}],
  "splits": {"train_test_split": {"test_size": 0.2, "random_state": 42}},
  "models": {
    "Random Forest Regression": {
      "parameters": {
        "n_estimators": 10,
        "max_depth": 5
      }
    }
  },
  "saved_models": {"dir": "models"},
  "reports": {"csv": {"dir": "reports", "name": "rf_report"}},
  "plots": {"dir": "plots"}
}
```

### 2. Implementation Example
```python
import json
from ml_fundamentals.chapter1.src.RandomForestRegression import RandomForestRegression

# 1. Load config
with open('ml_fundamentals/chapter1/configs/regression.json') as f:
    config = json.load(f)

# 2. Instantiate Model
rf = RandomForestRegression(config)

# 3. Load and Preprocess Data
rf.load_data()
rf.preprocess_data()

# 4. Train Model
rf.train()

# 5. Predict and Evaluate
y_pred = rf.predict(rf.X_test)
metrics = rf.evaluate(rf.y_test, y_pred)
print("Metrics:", metrics)

# 6. Save Model and Generate Outputs
rf.save_model()
rf.generate_report(metrics)
rf.generate_plots(rf.y_test, y_pred)

---
**File Location:** `ml_fundamentals/chapter1/src/RandomForestRegression.py`
```
