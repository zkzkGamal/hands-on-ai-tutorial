# Linear Regression

Linear Regression is a fundamental supervised machine learning algorithm used for predicting a continuous target variable based on one or more input features. It assumes a linear relationship between the inputs and the output.

## Mathematical Intuition

### 1. Hypothesis
The model predicts the target variable `y` using a linear combination of the input features `X`:
$$ \hat{y} = X \cdot \theta $$
where:
- `X` is the feature matrix (with a column of 1s for the intercept if applicable).
- `\theta` (theta) represents the learned weights or coefficients.

### 2. Cost Function
The objective is to find the weights `\theta` that minimize the Mean Squared Error (MSE) between the predicted values `\hat{y}` and the actual values `y`:
$$ J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 $$

### 3. Normal Equation (Closed-form Solution)
Instead of using iterative optimization algorithms like Gradient Descent, our implementation uses the **Normal Equation**. This approach finds the optimal weights directly by taking the derivative of the cost function with respect to `\theta` and setting it to zero:
$$ \theta = (X^T X)^{-1} X^T y $$
*Note: This requires computing the inverse of the matrix $(X^T X)$, which can be computationally expensive for very large datasets.*

---

## How to Use

The `LinearRegression` class in `src/LinearRegression.py` provides a full pipeline from data loading to model evaluation.

### 1. Configuration Setup
The class expects a configuration dictionary (typically loaded from a JSON file) that contains paths and parameters:
```json
{
  "datasets": [{"path": "data.csv", "target": "target_column"}],
  "splits": {"train_test_split": {"test_size": 0.2, "random_state": 42}},
  "saved_models": {"dir": "models", "names": ["linear_model.pkl"]},
  "reports": {"csv": {"dir": "reports", "name": "linear_report"}},
  "plots": {"dir": "plots", "names": ["linear_plot"]}
}
```

### 2. Implementation Example
```python
import json
from ml_fundamentals.chapter1.src.LinearRegression import LinearRegression

# 1. Load config
with open('ml_fundamentals/chapter1/configs/regression.json') as f:
    config = json.load(f)

# 2. Instantiate Model
model = LinearRegression(config)

# 3. Load and Preprocess Data
X, y = model.load_data()
X_train, X_test, y_train, y_test = model.preprocess_data()

# 4. Train Model
model.train()

# 5. Predict and Evaluate
y_pred = model.predict(model.X_test)
metrics = model.evaluate(model.y_test, y_pred)
print("Metrics:", metrics)

# 6. Save Model and Generate Outputs
model.save_model()
model.generate_report(metrics)
model.generate_plots()

---
**File Location:** `ml_fundamentals/chapter1/src/LinearRegression.py`
```
