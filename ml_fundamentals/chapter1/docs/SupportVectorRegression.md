# Support Vector Regression (SVR)

Support Vector Regression (SVR) is a powerful algorithm that extends Support Vector Machines (SVM) to regression problems. Unlike standard linear regression that tries to minimize the error between predicted and actual values, SVR attempts to fit the error within a certain threshold, while keeping the model as flat as possible.

## Mathematical Intuition

### 1. Epsilon-Tube ($\epsilon$-Tube)
The model constructs an $\epsilon$-insensitive tube around the predictions. Errors that fall within this tube are ignored, meaning no penalty is given for predictions that are "close enough" to the actual values. The objective is to ensure that all data points are as close as possible to the prediction line, but at the same time, we only penalize points that are further away than $\epsilon$.

### 2. Kernel Trick
To capture non-linear relationships, the inputs can be mapped into high-dimensional feature spaces using kernel functions. This allows the model to separate or fit data that isn't linearly separable in the original space. Supported kernels include:
- **Linear**: $K(x_i, x_j) = x_i \cdot x_j$
- **Polynomial**: $K(x_i, x_j) = (x_i \cdot x_j + 1)^d$
- **RBF (Gaussian)**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$

### 3. SMO Optimization
To find the optimal weights (alphas), the implementation uses a simplified Sequential Minimal Optimization (SMO) style algorithm. It iteratively updates individual alpha coefficients (Lagrange multipliers) based on the prediction error. If the error falls outside the $\epsilon$ boundary, the alphas are adjusted to reduce the error, strictly bounded by a regularization parameter $C$.

---

## How to Use

The `SupportVectorRegression` class in `src/SupportVectorRegression.py` provides a customizable SVR pipeline utilizing the SMO optimization technique.

### 1. Configuration Setup
The class expects a configuration detailing the kernel type, degree, gamma, epsilon, and learning rate:
```json
{
  "datasets": [{"path": "data.csv", "target": "target_column"}],
  "splits": {"train_test_split": {"test_size": 0.2, "random_state": 42}},
  "models": {
    "Support Vector Regression": {
      "parameters": {
        "kernel": "rbf",
        "degree": 3,
        "gamma": 0.1,
        "C": 1.0,
        "epsilon": 0.1,
        "learning_rate": 0.01,
        "epochs": 100
      }
    }
  },
  "saved_models": {"dir": "models"},
  "reports": {"csv": {"dir": "reports", "name": "svr_report"}},
  "plots": {"dir": "plots"}
}
```

### 2. Implementation Example
```python
import json
from ml_fundamentals.chapter1.src.SupportVectorRegression import SupportVectorRegression

# 1. Load config
with open('ml_fundamentals/chapter1/configs/regression.json') as f:
    config = json.load(f)

# 2. Instantiate Model
svr = SupportVectorRegression(config)

# 3. Load and Preprocess Data
svr.load_data()
svr.preprocess_data()

# 4. Train Model
svr.train()

# 5. Predict and Evaluate
y_pred = svr.predict(svr.X_test)
metrics = svr.evaluate(svr.y_test, y_pred)
print("Metrics:", metrics)

# 6. Save Model and Generate Outputs
svr.save_model()
svr.generate_report(metrics)
svr.generate_plots(svr.y_test, y_pred)

---
**File Location:** `ml_fundamentals/chapter1/src/SupportVectorRegression.py`
```
