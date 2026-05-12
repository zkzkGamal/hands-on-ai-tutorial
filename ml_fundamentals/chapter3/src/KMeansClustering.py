import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans as _SklearnKMeans   # used only by yellowbrick helpers


class KMeansClustering:
    """
    An implementation of K-Means Clustering built from scratch.

    K-Means partitions n observations into k clusters by iteratively assigning
    each point to the nearest centroid and then recomputing centroids as the
    mean of all points assigned to that cluster.

    Mathematical Intuition:
    1. **Objective**: Minimise the Within-Cluster Sum of Squares (WCSS / Inertia):
           J = Σ_k Σ_{x ∈ C_k} ‖x − μ_k‖²
       where μ_k is the centroid of cluster C_k.

    2. **Initialisation (k-means++)**: Rather than choosing centroids at random,
       each successive centroid is sampled with probability proportional to the
       squared distance from the nearest already-chosen centroid. This gives a
       O(log k) approximation guarantee and dramatically speeds up convergence.

    3. **Assignment step**: Each point x_i is assigned to the cluster whose
       centroid is closest:
           c_i = argmin_k ‖x_i − μ_k‖²

    4. **Update step**: Each centroid is recomputed as the mean of its members:
           μ_k = (1/|C_k|) Σ_{x ∈ C_k} x

    5. **Convergence**: Repeat steps 3–4 until centroids do not change (within a
       small tolerance) or max_iter is reached.  Because the algorithm may
       converge to a local minimum, it is run n_init times and the run with the
       lowest inertia is kept.

    Clustering Evaluation Metrics:
    - **Inertia (WCSS)**: Lower is better; measures compactness within clusters.
    - **Silhouette Score**: ∈ [-1, 1]; higher is better; measures how similar a
      point is to its own cluster vs. neighbouring clusters.
    - **Elbow Method**: Plot inertia vs. k; the "elbow" suggests the optimal k.

    References:
    - Lloyd, S. (1982). Least squares quantization in PCM. IEEE Transactions on
      Information Theory.
    - Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of careful
      seeding. SODA '07.
    - Scikit-Learn docs: https://scikit-learn.org/stable/modules/clustering.html#k-means
    """

    # ------------------------------------------------------------------ #
    #  Construction & configuration                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, config):
        """
        Initialise the KMeans Clustering model from a config dict.

        Parameters
        ----------
        config : dict
            Parsed clustering.json configuration.
        """
        self.config = config
        self.model_config = self.config['models']['K-Means Clustering']
        params = self.model_config['parameters']

        self.n_clusters   = params['n_clusters']
        self.init_method  = params.get('init', 'k-means++')
        self.n_init       = params.get('n_init', 10)
        self.max_iter     = params.get('max_iter', 300)
        self.random_state = params.get('random_state', 42)
        self.tol          = 1e-4   # convergence tolerance

        # State
        self.centroids      = None   # shape (n_clusters, n_features)
        self.labels_        = None   # cluster label for each training point
        self.inertia_       = None   # best WCSS found across n_init runs
        self.n_iter_        = None   # iterations used in the best run
        self.scaler         = StandardScaler()

        # Data holders
        self.X_raw       = None   # original (unscaled) feature DataFrame
        self.y_raw       = None   # optional ground-truth labels (not used for training)
        self.X_scaled    = None   # scaled numpy array used for training / prediction

    # ------------------------------------------------------------------ #
    #  Data pipeline                                                       #
    # ------------------------------------------------------------------ #

    def load_data(self, dataset_index: int = 1):
        """
        Load a dataset from the config.

        Parameters
        ----------
        dataset_index : int
            Index into config['datasets'] list (default 1 → penguins).

        Returns
        -------
        X_raw : pd.DataFrame   Feature columns.
        y_raw : pd.Series|None Ground-truth labels if a target column is defined.
        """
        dataset_info = self.config['datasets'][dataset_index]
        df = pd.read_csv(dataset_info['path'])

        target = dataset_info.get('target', '')
        if target and target in df.columns:
            self.y_raw = df.pop(target)
        else:
            self.y_raw = None

        # Drop rows with missing values and reset index
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        if self.y_raw is not None:
            self.y_raw = self.y_raw[df.index] if len(self.y_raw) != len(df) else self.y_raw
            self.y_raw.reset_index(drop=True, inplace=True)

        # Keep only numeric columns for clustering
        self.X_raw = df.select_dtypes(include=[np.number])
        print(f"Loaded dataset: '{dataset_info['name']}' | "
              f"Shape: {self.X_raw.shape} | "
              f"Target: '{target or 'None'}'")
        return self.X_raw, self.y_raw

    def preprocess_data(self):
        """
        Scale features to zero mean and unit variance.

        K-Means uses Euclidean distance, so feature scaling is essential to
        prevent high-magnitude features from dominating the distance computation.

        Returns
        -------
        X_scaled : np.ndarray   Scaled feature matrix.
        """
        self.X_scaled = self.scaler.fit_transform(self.X_raw)
        print(f"Features scaled: mean ≈ {self.X_scaled.mean():.4f}, "
              f"std ≈ {self.X_scaled.std():.4f}")
        return self.X_scaled

    # ------------------------------------------------------------------ #
    #  Core K-Means algorithm (from scratch)                              #
    # ------------------------------------------------------------------ #

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialise cluster centroids.

        Supports two strategies:
          - 'k-means++' : probabilistic seeding (Arthur & Vassilvitskii, 2007).
          - 'random'    : pick k random points from X.

        Parameters
        ----------
        X : np.ndarray  Shape (n_samples, n_features).

        Returns
        -------
        centroids : np.ndarray  Shape (n_clusters, n_features).
        """
        rng = np.random.RandomState(self.random_state)

        if self.init_method == 'k-means++':
            # Step 1: choose the first centroid uniformly at random
            idx = rng.randint(0, X.shape[0])
            centroids = [X[idx]]

            # Step 2: choose remaining centroids with D² weighting
            for _ in range(1, self.n_clusters):
                # Squared distance from each point to its nearest chosen centroid
                dists = np.array([
                    min(np.sum((x - c) ** 2) for c in centroids)
                    for x in X
                ])
                # Sample proportional to distance squared
                probs = dists / dists.sum()
                cumprobs = np.cumsum(probs)
                r = rng.rand()
                next_idx = np.searchsorted(cumprobs, r)
                centroids.append(X[next_idx])

            return np.array(centroids)

        else:  # 'random'
            indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices].copy()

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances from each point to each centroid.

        Parameters
        ----------
        X         : np.ndarray  Shape (n_samples, n_features).
        centroids : np.ndarray  Shape (n_clusters, n_features).

        Returns
        -------
        distances : np.ndarray  Shape (n_samples, n_clusters).
        """
        # Broadcasting: (n_samples, 1, n_features) - (1, n_clusters, n_features)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sum(diff ** 2, axis=2)   # shape: (n_samples, n_clusters)

    def _assign_clusters(self, distances: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the cluster with the nearest centroid.

        Parameters
        ----------
        distances : np.ndarray  Shape (n_samples, n_clusters).

        Returns
        -------
        labels : np.ndarray  Shape (n_samples,).
        """
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Recompute centroids as the mean of assigned points.

        If a cluster is empty (no points assigned), its centroid is re-initialised
        to a random point from X to avoid degenerate solutions.

        Parameters
        ----------
        X      : np.ndarray  Shape (n_samples, n_features).
        labels : np.ndarray  Shape (n_samples,).

        Returns
        -------
        new_centroids : np.ndarray  Shape (n_clusters, n_features).
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        rng = np.random.RandomState(self.random_state)

        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                # Empty cluster — reinitialise to a random data point
                new_centroids[k] = X[rng.randint(0, X.shape[0])]
            else:
                new_centroids[k] = members.mean(axis=0)

        return new_centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                         centroids: np.ndarray) -> float:
        """
        Compute inertia (WCSS) for a given assignment.

        Inertia = Σ_k Σ_{x ∈ C_k} ‖x − μ_k‖²

        Parameters
        ----------
        X         : np.ndarray  Shape (n_samples, n_features).
        labels    : np.ndarray  Shape (n_samples,).
        centroids : np.ndarray  Shape (n_clusters, n_features).

        Returns
        -------
        inertia : float
        """
        return float(sum(
            np.sum((X[labels == k] - centroids[k]) ** 2)
            for k in range(self.n_clusters)
        ))

    def _run_single(self, X: np.ndarray):
        """
        Run one full K-Means trial (init → assign → update loop).

        Parameters
        ----------
        X : np.ndarray  Shape (n_samples, n_features).

        Returns
        -------
        centroids : np.ndarray  Best centroids found.
        labels    : np.ndarray  Cluster labels.
        inertia   : float       WCSS for this trial.
        n_iter    : int         Number of iterations until convergence.
        """
        centroids = self._init_centroids(X)

        for iteration in range(1, self.max_iter + 1):
            distances = self._compute_distances(X, centroids)
            labels    = self._assign_clusters(distances)
            new_centroids = self._update_centroids(X, labels)

            # Convergence check: shift in centroid positions
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break

        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def train(self, X: np.ndarray = None):
        """
        Fit K-Means to the data.

        Runs n_init independent trials and keeps the solution with the lowest
        inertia (WCSS), mimicking scikit-learn's behaviour.

        Parameters
        ----------
        X : np.ndarray, optional
            Scaled feature matrix.  Defaults to self.X_scaled.

        Returns
        -------
        self
        """
        if X is None:
            X = self.X_scaled

        best_inertia   = np.inf
        best_centroids = None
        best_labels    = None
        best_n_iter    = None

        print(f"\nTraining K-Means (k={self.n_clusters}, "
              f"init='{self.init_method}', n_init={self.n_init}) ...")

        for run in range(self.n_init):
            centroids, labels, inertia, n_iter = self._run_single(X)
            print(f"  Run {run + 1:2d}/{self.n_init} | "
                  f"Inertia: {inertia:,.2f} | Iters: {n_iter}")

            if inertia < best_inertia:
                best_inertia   = inertia
                best_centroids = centroids
                best_labels    = labels
                best_n_iter    = n_iter

        self.centroids  = best_centroids
        self.labels_    = best_labels
        self.inertia_   = best_inertia
        self.n_iter_    = best_n_iter

        print(f"\nBest inertia: {self.inertia_:,.2f} (converged in {self.n_iter_} iter)")
        return self

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """
        Assign new data points to the nearest centroid.

        Parameters
        ----------
        X : np.ndarray, optional
            Scaled feature matrix.  Defaults to self.X_scaled.

        Returns
        -------
        labels : np.ndarray  Shape (n_samples,).
        """
        if X is None:
            X = self.X_scaled
        if self.centroids is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_model(self):
        """Save centroids, scaler, and metadata using joblib."""
        save_path = (
            f"{self.config['saved_models']['dir']}/"
            f"{self.config['saved_models']['names']['K-Means Clustering']}"
        )
        payload = {
            'centroids':  self.centroids,
            'scaler':     self.scaler,
            'n_clusters': self.n_clusters,
            'inertia':    self.inertia_,
            'n_iter':     self.n_iter_,
        }
        joblib.dump(payload, save_path)
        print(f"Model saved → {save_path}")

    def load_model(self):
        """Load a previously saved model."""
        load_path = (
            f"{self.config['saved_models']['dir']}/"
            f"{self.config['saved_models']['names']['K-Means Clustering']}"
        )
        payload = joblib.load(load_path)
        self.centroids  = payload['centroids']
        self.scaler     = payload['scaler']
        self.n_clusters = payload['n_clusters']
        self.inertia_   = payload['inertia']
        self.n_iter_    = payload['n_iter']
        print(f"Model loaded ← {load_path}")

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    def generate_report(self, metrics: dict):
        """
        Save clustering metrics to a JSON file.

        Parameters
        ----------
        metrics : dict  e.g. {'silhouette': 0.55, 'inertia': 123.4, ...}
        """
        import json as _json, os

        report_dir = self.config['plots']['dir']   # reuse plots dir
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/kmeans_report.json"

        with open(report_path, 'w') as f:
            _json.dump(metrics, f, indent=4)

        print(f"\n--- K-Means Report ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"Report saved → {report_path}")

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def generate_plots(self, X_scaled: np.ndarray = None):
        """
        Generate and save all diagnostic plots:

        1. **Elbow Method** – Inertia vs. k (uses yellowbrick KElbowVisualizer).
        2. **Silhouette Plot** – Per-cluster silhouette analysis.
        3. **PCA Scatter**    – 2-D projection coloured by cluster label.
        4. **Cluster Distribution** – Bar chart of cluster sizes.

        Parameters
        ----------
        X_scaled : np.ndarray, optional  Defaults to self.X_scaled.
        """
        import os

        if X_scaled is None:
            X_scaled = self.X_scaled

        plots_dir = self.config['plots']['dir']
        names     = self.config['plots']['names']
        os.makedirs(plots_dir, exist_ok=True)

        # ── 1. Elbow Method ──────────────────────────────────────────────
        print("Plotting elbow method ...")
        fig, ax = plt.subplots(figsize=(8, 5))
        km_proxy = _SklearnKMeans(init=self.init_method,
                                  n_init=self.n_init,
                                  random_state=self.random_state)
        visualizer = KElbowVisualizer(km_proxy, k=(2, 10), ax=ax)
        visualizer.fit(X_scaled)
        visualizer.finalize()
        ax.set_title('Elbow Method — Optimal k', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{names['elbow_method']}_kmeans.png", dpi=150)
        plt.close()

        # ── 2. Silhouette Analysis ───────────────────────────────────────
        print("Plotting silhouette analysis ...")
        fig, ax = plt.subplots(figsize=(8, 6))
        km_proxy2 = _SklearnKMeans(n_clusters=self.n_clusters,
                                   init=self.init_method,
                                   n_init=self.n_init,
                                   random_state=self.random_state)
        sil_viz = SilhouetteVisualizer(km_proxy2, colors='yellowbrick', ax=ax)
        sil_viz.fit(X_scaled)
        sil_viz.finalize()
        ax.set_title(f'Silhouette Analysis (k={self.n_clusters})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{names['silhouette_scores']}_kmeans.png", dpi=150)
        plt.close()

        # ── 3. PCA 2-D Cluster Scatter ───────────────────────────────────
        print("Plotting PCA cluster scatter ...")
        pca = PCA(n_components=2, random_state=self.random_state)
        X_2d = pca.fit_transform(X_scaled)

        palette = sns.color_palette('tab10', self.n_clusters)
        fig, ax = plt.subplots(figsize=(8, 6))

        for k in range(self.n_clusters):
            mask = self.labels_ == k
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[palette[k]], label=f'Cluster {k}',
                       alpha=0.7, edgecolors='white', linewidths=0.4, s=60)

        # Plot centroids projected into PCA space
        centroids_2d = pca.transform(self.centroids)
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   c='black', marker='X', s=180, zorder=5, label='Centroids')

        ax.set_title(f'K-Means Clusters — PCA Projection (k={self.n_clusters})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{names['cluster_centers']}_kmeans.png", dpi=150)
        plt.close()

        # ── 4. Cluster Size Distribution ────────────────────────────────
        print("Plotting cluster distribution ...")
        unique, counts = np.unique(self.labels_, return_counts=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar([f'Cluster {k}' for k in unique], counts,
                      color=[palette[k] for k in unique],
                      edgecolor='white', linewidth=1.2)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, str(count),
                    ha='center', va='bottom', fontweight='bold')
        ax.set_title(f'Cluster Size Distribution (k={self.n_clusters})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Points')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{names['cluster_distribution']}_kmeans.png", dpi=150)
        plt.close()

        print(f"\nAll plots saved to → {plots_dir}/")


# ====================================================================== #
#  Entry-point                                                            #
# ====================================================================== #

if __name__ == "__main__":
    import json, os
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
    )

    # ── Load config ──────────────────────────────────────────────────────
    config = json.load(open('ml_fundamentals/chapter3/configs/clustering.json'))

    os.makedirs(config['saved_models']['dir'], exist_ok=True)
    os.makedirs(config['plots']['dir'],        exist_ok=True)

    # ── Build & train model ──────────────────────────────────────────────
    km = KMeansClustering(config)
    km.load_data(dataset_index=0)       # glass dataset
    km.preprocess_data()
    km.train()

    # ── Evaluate ─────────────────────────────────────────────────────────
    labels   = km.labels_
    X_scaled = km.X_scaled

    metrics = {
        "n_clusters":        km.n_clusters,
        "inertia":           round(km.inertia_, 4),
        "silhouette_score":  round(silhouette_score(X_scaled, labels), 4),
        "davies_bouldin":    round(davies_bouldin_score(X_scaled, labels), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 4),
        "n_iterations":      km.n_iter_,
    }

    km.generate_report(metrics)

    # ── Persist & visualise ──────────────────────────────────────────────
    km.save_model()
    km.generate_plots()
