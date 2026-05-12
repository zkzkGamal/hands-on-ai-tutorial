# Mini-Batch K-Means Clustering

## Overview

Mini-Batch K-Means is a scalable variant of the standard K-Means algorithm designed for large datasets. Instead of using the entire dataset at each iteration, it processes small random batches of data, making it much faster while maintaining similar clustering quality.

This implementation is built **from scratch** using NumPy, demonstrating the online learning approach.

## Mathematical Foundation

### Core Algorithm
Mini-Batch K-Means modifies the standard K-Means update rule to use **online stochastic gradient descent**:

1. **Mini-batch Sampling**: At each iteration, sample a batch B of size `batch_size`
2. **Assignment**: Assign batch points to nearest centroids
3. **Online Update**: Update centroids using a decreasing learning rate

### Online Centroid Update
For each point x in the mini-batch assigned to cluster k:

```
n_k ← n_k + 1                    # Running count of points seen by cluster k
η ← 1/n_k                        # Learning rate (decreases over time)
μ_k ← (1 - η) × μ_k + η × x      # Exponential moving average update
```

This creates an **adaptive learning rate** that naturally down-weights older observations.

### Convergence
The algorithm converges when the maximum centroid shift between iterations falls below a tolerance:

```
shift = ||μ_new - μ_old||_2
if shift < tolerance: break
```

## Code Structure

### Class: `MiniBatchKMeansClustering`

#### Key Methods

- `__init__(config)`: Initialize with batch_size and other parameters
- `load_data(dataset_index)`: Load dataset from configuration
- `preprocess_data()`: Feature standardization
- `train(X)`: Fit model with multiple initializations
- `predict(X)`: Assign new points to clusters
- `save_model()` / `load_model()`: Model persistence
- `generate_report(metrics)`: Save evaluation results
- `generate_plots()`: Create diagnostic visualizations

#### Core Algorithm Methods

- `_init_centroids(X)`: K-means++ or random initialization
- `_compute_distances(X, centroids)`: Distance matrix computation
- `_assign_clusters(distances)`: Cluster assignment
- `_run_single(X)`: Single training run with mini-batch updates

## Key Features

### Mini-Batch Processing
```python
# Sample random mini-batch without replacement
batch_idx = rng.choice(X.shape[0],
                       size=min(self.batch_size, X.shape[0]),
                       replace=False)
X_batch = X[batch_idx]
```

### Online Learning Update
```python
for i, k in enumerate(batch_labels):
    counts[k] += 1
    eta = 1.0 / counts[k]  # Decreasing learning rate
    centroids[k] = (1 - eta) * centroids[k] + eta * X_batch[i]
```

### Multiple Initializations
Like standard K-Means, runs `n_init` independent trials to avoid local minima.

## Usage Example

```python
import json
from src.MiniBatchKMeansClustering import MiniBatchKMeansClustering

# Load configuration
config = json.load(open('configs/clustering.json'))

# Create and train model
mbkm = MiniBatchKMeansClustering(config)
mbkm.load_data(dataset_index=0)  # Load glass dataset
mbkm.preprocess_data()
mbkm.train()

# Evaluate clustering quality
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(mbkm.X_scaled, mbkm.labels_)
print(f"Silhouette Score: {silhouette:.3f}")

# Generate visualizations and save
mbkm.generate_plots()
mbkm.save_model()
```

## Advantages over Standard K-Means

### Computational Efficiency
- **Time Complexity**: O(batch_size × K × d) per iteration vs O(N × K × d)
- **Memory Usage**: Only batch_size points in memory at once
- **Scalability**: Handles datasets with millions of points

### Practical Benefits
- **Streaming Data**: Can process data in chunks
- **Large Datasets**: Suitable for big data applications
- **Parallelization**: Easier to parallelize across batches

## Trade-offs

### Quality vs Speed
- **Higher Inertia**: Typically produces slightly looser clusters than full K-Means
- **Noisier Convergence**: Stochastic updates can cause more variable results
- **Parameter Sensitivity**: Performance depends on batch_size choice

### Batch Size Selection
- **Small batches**: Faster but noisier convergence
- **Large batches**: Slower but more stable (approaches full K-Means)
- **Rule of thumb**: batch_size = 100-1000, or 0.1-1% of dataset size

## Mathematical Properties

### Learning Rate Schedule
The learning rate η = 1/n_k creates an **inverse time decay** schedule that ensures convergence:

- Early iterations: High learning rate for rapid adaptation
- Later iterations: Low learning rate for fine-tuning

### Convergence Guarantees
While not as theoretically sound as full K-Means, mini-batch K-Means:
- Converges to a local minimum of the WCSS objective
- Achieves similar clustering quality with significant speed gains
- Maintains the same O(log K) approximation guarantee with k-means++ initialization

## Evaluation Metrics

### Same as Standard K-Means
- **Inertia (WCSS)**: Within-cluster sum of squares
- **Silhouette Score**: Cluster separation and cohesion measure
- **Calinski-Harabasz Index**: Ratio of between/within cluster variance

### Performance Comparison
When comparing to standard K-Means:
- **Speed**: 5-10x faster on large datasets
- **Quality**: 95-99% of full K-Means performance
- **Memory**: Constant memory usage regardless of dataset size

## Visualizations

Generates the same diagnostic plots as standard K-Means:
1. Elbow method for optimal K selection
2. Silhouette analysis per cluster
3. PCA scatter plot with cluster colors
4. Cluster size distribution

All plots include "_minibatch" suffix to distinguish from standard K-Means results.

## Applications

### When to Use Mini-Batch K-Means

- **Large Datasets**: Millions of data points
- **Streaming Data**: Continuous data streams
- **Limited Memory**: Cannot fit entire dataset in RAM
- **Real-time Clustering**: Need fast updates
- **Big Data**: Distributed computing environments

### Example Use Cases
- Customer segmentation for e-commerce platforms
- Log analysis for cybersecurity
- Image clustering in computer vision
- Document clustering for text mining

## Implementation Notes

### Distance Computation
Uses the same efficient broadcasting approach as standard K-Means:

```python
diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
distances = np.sum(diff ** 2, axis=2)  # Shape: (n_samples, n_clusters)
```

### Random State Management
Proper random seeding ensures reproducible results across multiple runs.

### Error Handling
- Checks for empty clusters during centroid updates
- Handles edge cases like batch_size > dataset_size

## References

- Sculley, D. (2010). Web-scale k-means clustering. WWW '10
- Scikit-Learn Documentation: https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans

## Related Algorithms

- **Standard K-Means**: Full-batch version for smaller datasets
- **Online K-Means**: Similar but processes one point at a time
- **Streaming K-Means**: Designed specifically for data streams