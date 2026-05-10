# Logistic Regression

Logistic Regression is a foundational classification algorithm used to predict the probability of a categorical dependent variable. Despite its name, it is used for classification, not regression.

## Mathematical Intuition

### 1. Hypothesis (Sigmoid Function)
The model calculates a linear combination of inputs and then passes the result through the Sigmoid function to map it to a probability between 0 and 1:
$$ z = X \cdot w + b $$
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
where:
- `X` is the feature matrix.
- `w` represents the learned weights.
- `b` is the bias term.

### 2. Cost Function (Binary Cross-Entropy)
To evaluate the model, we use the Binary Cross-Entropy (Log Loss) cost function, which measures the difference between the predicted probabilities and the true labels:
$$ J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] $$

### 3. Gradient Descent
We minimize the cost function iteratively using Gradient Descent, updating the weights and bias based on the gradients:
$$ w = w - \alpha \frac{\partial J}{\partial w} $$
$$ b = b - \alpha \frac{\partial J}{\partial b} $$
where $\alpha$ is the learning rate.

---

## How to Use

The `LogisticRegression` class in `src/LogisticRegression.py` provides a full pipeline from data loading to model evaluation.

### Implementation Example
```python
import json
from ml_fundamentals.chapter2.src.LogisticRegression import LogisticRegression

# 1. Load config
with open('ml_fundamentals/chapter2/configs/classification.json') as f:
    config = json.load(f)

# 2. Instantiate Model
model = LogisticRegression(config)

# 3. Load and Preprocess Data
model.load_data()
model.preprocess_data()

# 4. Train Model
model.train(model.X_train, model.y_train)

# 5. Predict
y_pred = model.predict(model.X_test)

# 6. Save Model and Generate Outputs
model.save_model()
model.generate_report({"accuracy": 0.95}) # Example metric
model.generate_plots()
```

---
**File Location:** `ml_fundamentals/chapter2/src/LogisticRegression.py`
