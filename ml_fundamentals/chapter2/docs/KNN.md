# K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a non-parametric, lazy learning algorithm used for classification. It classifies a new data point based on the majority class among its `k` closest neighbors in the training set.

## Mathematical Intuition

### 1. Distance Metric
To find the closest neighbors, the algorithm computes the distance between the query point and all points in the training set. Euclidean distance is typically used:
$$ d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2} $$

### 2. Vectorized Distance Calculation
For efficiency, our implementation uses a heavily vectorized approach to compute distances across all points simultaneously without loops. Utilizing the expansion $(a-b)^2 = a^2 - 2ab + b^2$:
$$ d(X_{test}, X_{train}) = \sqrt{X_{test}^2 - 2 X_{test} X_{train}^T + (X_{train}^2)^T} $$

### 3. Majority Voting
Once the distances are computed, the algorithm sorts them to find the `k` nearest neighbors. The predicted class is the mode (most frequent class) among these `k` neighbors.

---

## How to Use

The `KNearestNeighbor` class in `src/KNN.py` provides a full pipeline from data loading to model evaluation.

### Implementation Example
```python
import json
from ml_fundamentals.chapter2.src.KNN import KNearestNeighbor

# 1. Load config
with open('ml_fundamentals/chapter2/configs/classification.json') as f:
    config = json.load(f)

# 2. Instantiate Model
model = KNearestNeighbor(config)

# 3. Load and Preprocess Data
model.load_data()
model.preprocess_data()

# 4. "Train" Model (KNN just stores the data)
model.train(model.X_train, model.y_train)

# 5. Predict
y_pred = model.predict(model.X_test)

# 6. Save Model and Generate Outputs
model.save_model()
model.generate_report({"accuracy": 0.95}) # Example metric
model.generate_plots()
```

---
**File Location:** `ml_fundamentals/chapter2/src/KNN.py`
