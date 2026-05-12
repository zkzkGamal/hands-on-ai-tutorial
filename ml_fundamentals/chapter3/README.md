<div align="center">

<h1>🔍 Chapter 3: Clustering</h1>

<p><strong>Discover hidden patterns in data — group similar items without labels.</strong></p>

<p><em>Part of the <a href="../../README.md">Hands-on AI Tutorial</a> · Machine Learning Fundamentals Track</em></p>

<br/>

[![Track](https://img.shields.io/badge/Track-ML%20Fundamentals-blueviolet?style=flat-square)]()
[![Chapter](https://img.shields.io/badge/Chapter-3%20of%205-blue?style=flat-square)]()
[![Difficulty](https://img.shields.io/badge/Level-Intermediate-brightgreen?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()

<br/>

| 🧮 Algorithms | 📊 Datasets | 📏 Metrics | 📓 Notebooks |
|:-------------:|:-----------:|:----------:|:------------:|
| **6** | **2** | Silhouette · DBI | **1** |

<br/>

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

<br/>

**[🧩 What is Clustering?](#-what-is-clustering-for-beginners)** · **[🧮 Algorithms](#-algorithms-covered)** · **[📊 Datasets](#-datasets)** · **[🚀 How to Run](#-how-to-run)** · **[🗺️ Learning Path](#️-learning-path-for-this-chapter)**

<br/>

---

> *"Clustering is the art of finding structure in chaos."*

</div>

---

## 🧩 What Is Clustering? (For Beginners)

Imagine you have a bunch of customers and their shopping data, but no labels saying "loyal customer" or "one-time buyer". Clustering helps you automatically group similar customers together based on their behavior.

In simple terms:

> **Clustering = Finding natural groups in data without any supervision.**

Examples of clustering problems:
- 🛒 Customer segmentation for marketing
- 📸 Grouping similar images or documents
- 🧬 Finding patterns in genetic data
- 🌐 Detecting communities in social networks

Unlike classification (Chapter 2), clustering is **unsupervised** — there are no "correct" answers to learn from!

---

## 🎯 What You Will Learn

- ✅ What clustering is and how it differs from supervised learning
- ✅ The math behind 6 major clustering algorithms, including matrix operations
- ✅ How to choose the right number of clusters (K)
- ✅ How to evaluate clusters when you have no ground truth labels
- ✅ Understanding distance matrices, adjacency matrices, and similarity measures
- ✅ How to visualize high-dimensional clusters in 2D

---

## 🧮 Algorithms Covered

### 1. 🎯 K-Means Clustering
Partitions data into K clusters by minimizing the sum of squared distances from points to their cluster centroids.

- **Best for**: Spherical, well-separated clusters of similar sizes.
- **Key concept**: Iterative centroid updates and assignment.
- **Math**: Minimize `J = Σ_{i=1}^n ||x_i - μ_{c_i}||^2` where μ is the centroid.
- **Matrix**: Distance matrix between points and centroids.
- 📄 [Full Documentation](docs/KMeansClustering.md)

---

### 2. 🚀 Mini-Batch K-Means
A faster variant of K-Means that uses small random batches of data for each iteration.

- **Best for**: Large datasets where full K-Means is too slow.
- **Key concept**: Stochastic gradient descent on the K-Means objective.
- **Math**: Same as K-Means but with mini-batches: `μ = (1/|B|) Σ_{x∈B} x`
- **Matrix**: Same distance computations but on subsets.
- 📄 [Full Documentation](docs/MiniBatchKMeansClustering.md)
- 📄 [Full Documentation](docs/MiniBatchKMeansClustering.md)

---

### 3. 🌳 Agglomerative Hierarchical Clustering
Builds a hierarchy of clusters by successively merging the closest pairs.

- **Best for**: Understanding cluster relationships and hierarchies.
- **Key concept**: Linkage methods (single, complete, average) and distance matrices.
- **Math**: Distance between clusters: `d(C1,C2) = min/max/avg {d(x,y) | x∈C1, y∈C2}`
- **Matrix**: Full distance matrix between all points, updated during merging.

---

### 4. 🔗 DBSCAN (Density-Based Spatial Clustering)
Groups points that are closely packed together, marking outliers as noise.

- **Best for**: Arbitrary-shaped clusters and outlier detection.
- **Key concept**: Core points, density-reachability, and eps-neighborhoods.
- **Math**: A point is core if `|N_eps(p)| ≥ min_samples`, where N_eps is the eps-ball.
- **Matrix**: Adjacency matrix for density connections.

---

### 5. 📡 OPTICS (Ordering Points To Identify Clustering Structure)
An extension of DBSCAN that creates an ordering of points to find clusters at multiple density levels.

- **Best for**: Finding clusters in data with varying densities.
- **Key concept**: Core distance, reachability distance, and cluster ordering.
- **Math**: Reachability distance: `reachability(p,o) = max(core_dist(p), dist(p,o))`
- **Matrix**: Reachability plot and distance matrix.

---

### 6. 📈 Mean Shift Clustering
Finds cluster centers by iteratively shifting points towards regions of higher density.

- **Best for**: Non-parametric clustering without assuming cluster shapes.
- **Key concept**: Kernel density estimation and mode-seeking.
- **Math**: Shift vector: `m(x) = Σ_{x_i ∈ K(x)} (x_i - x) * K'(||x_i - x||/h)`
- **Matrix**: Kernel matrix for density estimation.

---

## 📊 Datasets

All datasets are stored in `data/` and referenced in the config file.

| Dataset | File | Description |
|---|---|---|
| Penguins | `penguins.csv` | Physical measurements of penguins from different species |
| Seed | `seed_dataset.csv` | Geometric properties of wheat kernels |

---

## 📏 Evaluation Metrics

Since clustering is unsupervised, evaluation uses internal measures of cluster quality:

| Metric | Formula / Meaning | What It Means |
|---|---|---|
| **Silhouette Score** | `(b - a) / max(a, b)` where a is intra-cluster distance, b is inter-cluster distance | Measures how similar an object is to its own cluster vs other clusters. Range: [-1, 1], higher is better. |
| **Davies-Bouldin Index** | Average of `max(R_ij + R_ji)/(M_ij)` for all cluster pairs | Measures average similarity of each cluster with its most similar cluster. Lower is better. |
| **Calinski-Harabasz Index** | `(SSB / (k-1)) / (SSW / (n-k))` where SSB is between-cluster variance, SSW is within-cluster variance | Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better. |

### 🧠 Deep Dive: The Mathematics of Clustering Evaluation

Clustering evaluation relies heavily on **distance matrices** and **variance calculations**:

- **Distance Matrix**: A symmetric matrix D where D_ij = distance(x_i, x_j). Used in hierarchical clustering and silhouette calculations.
- **Within-Cluster Sum of Squares (SSW)**: `Σ_{c} Σ_{x∈c} ||x - μ_c||^2`
- **Between-Cluster Sum of Squares (SSB)**: `Σ_{c} |c| * ||μ_c - μ_global||^2`

These metrics help quantify cluster cohesion (points in same cluster are close) and separation (points in different clusters are far apart).

---

## 🗂️ File Structure

```text
chapter3/
│
├── README.md                          ← You are here
│
├── configs/
│   └── clustering.json                ← Dataset paths, model hyperparameters
│
├── data/
│   ├── penguins.csv                   ← Penguin measurements dataset
│   └── seed_dataset.csv               ← Wheat kernel dataset
│
├── docs/
│   ├── KMeansClustering.md            ← Detailed K-Means documentation
│   ├── MiniBatchKMeansClustering.md   ← Mini-Batch K-Means documentation
│   └── Clustering.ipynb.md            ← Notebook exploration guide
│
├── models/
│   ├── kmeans.pkl                     ← Saved K-Means model
│   ├── minibatch_kmeans.pkl           ← Saved Mini-Batch K-Means
│   └── optics.pkl                     ← Saved OPTICS model
│
├── notebooks/
│   └── Clustering.ipynb               ← Interactive exploration notebook
│                                      📄 [Full Documentation](docs/Clustering.ipynb.md)
│
├── plots/
│   ├── elbow_method.png               ← Elbow method for optimal K
│   ├── silhouette_scores.png          ← Silhouette analysis plots
│   └── cluster_centers.png            ← Cluster visualization
│
└── src/
    ├── KMeansClustering.py            ← K-Means from scratch
    └── MiniBatchKMeansClustering.py   ← Mini-Batch K-Means from scratch
```

---

## ⚙️ Configuration File (`configs/clustering.json`)

The config file controls everything — datasets, model parameters, file paths. Here's what it contains:

```json
{
  "datasets": [...],        // List of datasets with file paths
  "models": {...},          // Each model's hyperparameters (e.g., n_clusters for K-Means, eps for DBSCAN)
  "metrics": {...},         // Which metrics to compute
  "plots": {...},           // Where to save visualizations
  "saved_models": {...}     // Where to save trained model files
}
```

---

## 🚀 How to Run

### Option A: Interactive (Recommended for Learners)

Open the Jupyter notebook to explore interactively:

```bash
jupyter notebook ml_fundamentals/chapter3/notebooks/Clustering.ipynb
```

---

### Option B: Run a Script Directly

Each model can be run as a standalone Python script from the **project root**:

```bash
# K-Means Clustering
python3 ml_fundamentals/chapter3/src/KMeansClustering.py

# Mini-Batch K-Means
python3 ml_fundamentals/chapter3/src/MiniBatchKMeansClustering.py
```

> ⚠️ Run from the **root** of the repository, not inside the chapter folder.

---

## 📋 Prerequisites

Make sure you have these Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## 🗺️ Learning Path for This Chapter

```text
Step 1: Read this README (you're doing it now ✅)

Step 2: Open the Notebook
  └── ml_fundamentals/chapter3/notebooks/Clustering.ipynb
      ├── Load the datasets
      ├── Explore distributions and correlations
      └── See algorithms in action visually

Step 3: Understand the Code
  └── See how the math translates to Python code in src/
      ├── Start with KMeansClustering.py
      └── See matrix operations for distance calculations

Step 4: Experiment!
  ├── Change 'n_clusters' for K-Means in clustering.json and see metrics change!
  ├── Switch to the seed dataset
  └── Compare different linkage methods in hierarchical clustering
```

---

## 💡 Tips for Beginners

- **Scale your data.** Clustering algorithms are sensitive to feature scales. Always standardize features with different units.
- **Choose K wisely.** Use the elbow method or silhouette analysis to find the optimal number of clusters.
- **Understand your distance metric.** Euclidean distance assumes spherical clusters; consider Manhattan or cosine distance for other shapes.
- **Watch for outliers.** Density-based methods like DBSCAN are great at identifying noise points.

---

## ➡️ What's Next?

Once you're comfortable with clustering, move on to:

**→ Chapter 4: Deep Learning Fundamentals** *(coming soon)*

You'll learn about neural networks and how they power modern AI.

---

> 📖 **Chapter 3 is part of the [Hands-on AI Tutorial](../../README.md)** — an open-source project for learning AI from scratch.
