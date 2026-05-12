# K-Means Clustering

## Overview

K-Means is one of the most popular and widely-used clustering algorithms in machine learning. It partitions a dataset into **K** distinct, non-overlapping clusters where each data point belongs to the cluster with the nearest mean (centroid).

This implementation is built **from scratch** using only NumPy, providing complete control over the algorithm and educational value.

## Mathematical Foundation

### Objective Function
K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also known as **inertia**:

```
J = Σ_{k=1}^K Σ_{x ∈ C_k} ||x - μ_k||²
```

Where:
- `K` is the number of clusters
- `C_k` is the set of points in cluster k
- `μ_k` is the centroid (mean) of cluster k
- `||x - μ_k||²` is the squared Euclidean distance

### Algorithm Steps

1. **Initialization**: Choose K initial centroids
   - **Random**: Select K random data points
   - **K-means++**: Probabilistic initialization for better convergence

2. **Assignment Step**: Assign each point to the nearest centroid
   ```
   c_i = argmin_k ||x_i - μ_k||²
   ```

3. **Update Step**: Recalculate centroids as the mean of assigned points
   ```
   μ_k = (1/|C_k|) Σ_{x ∈ C_k} x
   ```

4. **Convergence**: Repeat steps 2-3 until centroids don't change significantly

### Distance Matrix
The core computation involves a **distance matrix** D of shape (n_samples, n_clusters):

```
D[i,k] = Σ_{j=1}^d (x_i,j - μ_k,j)²
```

This matrix enables efficient assignment of all points to their nearest centroids.

## Code Structure

### Class: `KMeansClustering`

#### Key Methods

- `__init__(config)`: Initialize with configuration parameters
- `load_data(dataset_index)`: Load and preprocess dataset
- `preprocess_data()`: Standardize features using StandardScaler
- `train(X)`: Fit the model using multiple random initializations
- `predict(X)`: Assign new points to clusters
- `save_model()` / `load_model()`: Persistence with joblib
- `generate_report(metrics)`: Save evaluation metrics to JSON
- `generate_plots()`: Create diagnostic visualizations

#### Core Algorithm Methods

- `_init_centroids(X)`: Initialize centroids (random or k-means++)
- `_compute_distances(X, centroids)`: Calculate distance matrix
- `_assign_clusters(distances)`: Assign points to nearest centroids
- `_update_centroids(X, labels)`: Recalculate centroids
- `_compute_inertia(X, labels, centroids)`: Calculate WCSS
- `_run_single(X)`: Single K-Means run with convergence checking

## Key Features

### Initialization Strategies

**Random Initialization**:
```python
indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
centroids = X[indices].copy()
```

**K-means++ Initialization**:
```python
# First centroid: random selection
idx = rng.randint(0, X.shape[0])
centroids = [X[idx]]

# Subsequent centroids: probability ∝ distance² to nearest centroid
for _ in range(1, self.n_clusters):
    dists = [min(sum((x - c)**2) for c in centroids) for x in X]
    probs = dists / sum(dists)
    next_idx = np.searchsorted(np.cumsum(probs), rng.rand())
    centroids.append(X[next_idx])
```

### Convergence Criteria
The algorithm stops when the maximum centroid shift falls below a tolerance:

```python
shift = np.linalg.norm(new_centroids - centroids)
if shift < self.tol:
    break
```

### Multiple Initializations
To avoid local minima, the algorithm runs `n_init` times and selects the best solution:

```python
best_inertia = min(inertia for _, _, inertia, _ in runs)
```

## Usage Example

```python
import json
from src.KMeansClustering import KMeansClustering

# Load configuration
config = json.load(open('configs/clustering.json'))

# Create and train model
km = KMeansClustering(config)
km.load_data(dataset_index=1)  # Load penguins dataset
km.preprocess_data()
km.train()

# Evaluate
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(km.X_scaled, km.labels_)
print(f"Silhouette Score: {silhouette:.3f}")

# Visualize
km.generate_plots()
km.save_model()
```

## Evaluation Metrics

### Inertia (WCSS)
- Measures cluster compactness
- Lower values indicate tighter clusters
- Used in the elbow method for optimal K selection

### Silhouette Score
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
- `a(i)`: Average distance to other points in same cluster
- `b(i)`: Average distance to points in nearest other cluster
- Range: [-1, 1], higher values indicate better clustering

## Visualizations

The implementation generates four types of plots:

1. **Elbow Method**: Inertia vs. number of clusters
2. **Silhouette Analysis**: Per-cluster silhouette coefficients
3. **PCA Scatter Plot**: 2D projection with cluster colors
4. **Cluster Distribution**: Bar chart of cluster sizes

## Advantages & Limitations

### Advantages
- Simple and intuitive algorithm
- Scales well to large datasets
- Guaranteed convergence
- Easy to implement and understand

### Limitations
- Requires specifying K in advance
- Sensitive to initial centroid positions
- Assumes spherical, equally-sized clusters
- Struggles with non-convex cluster shapes
- Outliers can significantly affect centroids

## References

- Lloyd, S. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory
- Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. SODA '07
- Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/clustering.html#k-means

## Related Algorithms

- **Mini-Batch K-Means**: Scalable variant using data subsets
- **K-Medoids**: Uses actual data points as centroids (more robust to outliers)
- **Gaussian Mixture Models**: Probabilistic clustering with soft assignments