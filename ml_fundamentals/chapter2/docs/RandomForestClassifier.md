# Random Forest Classifier

A Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time. It combines the predictions of multiple trees to improve generalizability and robustness over a single decision tree, effectively combating overfitting.

## Mathematical Intuition

### 1. Bagging (Bootstrap Aggregating)
The algorithm trains each decision tree on a random sample of the dataset, drawn with replacement (a bootstrap sample). This introduces variations among the trees, reducing the variance of the overall ensemble.

### 2. Feature Randomness
When deciding where to split a node, a typical decision tree searches through all features. A Random Forest tree, however, only considers a random subset of features for each split. This decorrelates the individual trees, making the ensemble more powerful.

### 3. Aggregation (Majority Voting)
For classification tasks, the final prediction of the Random Forest is determined by a majority vote. Each individual tree makes a prediction, and the class that receives the most votes across all trees becomes the model's final output.

---

## How to Use

The `RandomForestClassifier` class in `src/RandomForestClassifier.py` provides a full pipeline and utilizes the custom `DecisionTreeClassifier` under the hood.

### Implementation Example
```python
import json
from ml_fundamentals.chapter2.src.RandomForestClassifier import RandomForestClassifier

# 1. Load config
with open('ml_fundamentals/chapter2/configs/classification.json') as f:
    config = json.load(f)

# 2. Instantiate Model
model = RandomForestClassifier(config)

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
**File Location:** `ml_fundamentals/chapter2/src/RandomForestClassifier.py`
