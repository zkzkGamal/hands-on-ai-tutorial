import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import MiniBatchKMeans as _SklearnMBKMeans  # used only by yellowbrick helpers


class MiniBatchKMeansClustering:
    """
    An implementation of Mini-Batch K-Means Clustering built from scratch.

    Mini-Batch K-Means is a scalable variant of standard K-Means that uses
    small random subsets (mini-batches) of the data at each iteration instead
    of the full dataset.  This drastically reduces computation time while
    converging to a solution close to the standard K-Means result.

    Mathematical Intuition:
    1. **Mini-batch sampling**: At each iteration t, draw a random batch B of
       size `batch_size` uniformly from X:
           B_t ⊂ X,   |B_t| = batch_size

    2. **Assignment step** (same as standard K-Means):
           c_i = argmin_k ‖x_i − μ_k‖²   for each x_i ∈ B_t

    3. **Online centroid update** (key difference from standard K-Means):
       For each x_i in the mini-batch assigned to cluster k:
           n_k  ← n_k + 1           (running count of samples seen by cluster k)
           η    = 1 / n_k           (per-sample learning rate, decreases over time)
           μ_k  ← (1 − η) · μ_k + η · x_i

       This is an exponential moving average that naturally down-weights old
       observations as more data arrives, ensuring convergence.

    4. **Convergence**: Measured by the maximum centroid shift between
       consecutive iterations.  The algorithm also runs n_init independent
       trials and keeps the one with the lowest inertia (WCSS).

    Advantages over Standard K-Means:
    - O(batch_size) instead of O(n_samples) per iteration → much faster.
    - Memory-efficient: only a batch is loaded at a time.
    - Well-suited for large-scale / streaming data.

    Trade-offs:
    - Slightly higher inertia (looser clusters) than full K-Means.
    - Convergence can be noisier due to stochastic updates.

    References:
    - Sculley, D. (2010). Web-scale k-means clustering. WWW '10.
    - Scikit-Learn docs: https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans
    """

    # ------------------------------------------------------------------ #
    #  Construction & configuration                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['models']['Mini-Batch K-Means Clustering']
        params = self.model_config['parameters']

        self.n_clusters   = params['n_clusters']
        self.init_method  = params.get('init', 'k-means++')
        self.n_init       = params.get('n_init', 10)
        self.max_iter     = params.get('max_iter', 300)
        self.random_state = params.get('random_state', 42)
        self.batch_size   = params.get('batch_size', 100)
        self.tol          = 1e-4

        # State
        self.centroids   = None   # shape (n_clusters, n_features)
        self.labels_     = None   # cluster label for every training point
        self.inertia_    = None   # best WCSS found across n_init runs
        self.n_iter_     = None   # iterations used in the best run
        self.scaler      = StandardScaler()

        # Data holders
        self.X_raw    = None
        self.y_raw    = None
        self.X_scaled = None

    # ------------------------------------------------------------------ #
    #  Data pipeline                                                       #
    # ------------------------------------------------------------------ #

    def load_data(self, dataset_index: int = 1):
        """Load a dataset specified in config['datasets']."""
        dataset_info = self.config['datasets'][dataset_index]
        df = pd.read_csv(dataset_info['path'])

        target = dataset_info.get('target', '')
        if target and target in df.columns:
            self.y_raw = df.pop(target)
        else:
            self.y_raw = None

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        if self.y_raw is not None:
            self.y_raw.reset_index(drop=True, inplace=True)

        self.X_raw = df.select_dtypes(include=[np.number])
        print(f"Loaded dataset: '{dataset_info['name']}' | "
              f"Shape: {self.X_raw.shape} | Target: '{target or 'None'}'")
        return self.X_raw, self.y_raw

    def preprocess_data(self):
        """Scale features to zero mean and unit variance."""
        self.X_scaled = self.scaler.fit_transform(self.X_raw)
        print(f"Features scaled: mean ≈ {self.X_scaled.mean():.4f}, "
              f"std ≈ {self.X_scaled.std():.4f}")
        return self.X_scaled

    # ------------------------------------------------------------------ #
    #  Core Mini-Batch K-Means algorithm (from scratch)                   #
    # ------------------------------------------------------------------ #

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialise centroids using k-means++ or random strategy.

        k-means++ samples the first centroid uniformly at random, then each
        subsequent centroid with probability ∝ D²(x) where D(x) is the
        distance to the nearest already-chosen centroid.
        """
        rng = np.random.RandomState(self.random_state)

        if self.init_method == 'k-means++':
            idx = rng.randint(0, X.shape[0])
            centroids = [X[idx]]
            for _ in range(1, self.n_clusters):
                dists = np.array([
                    min(np.sum((x - c) ** 2) for c in centroids)
                    for x in X
                ])
                probs = dists / dists.sum()
                r = rng.rand()
                next_idx = np.searchsorted(np.cumsum(probs), r)
                centroids.append(X[next_idx])
            return np.array(centroids)
        else:
            indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices].copy()

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Squared Euclidean distances via broadcasting.
        Returns shape (n_samples, n_clusters).
        """
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sum(diff ** 2, axis=2)

    def _assign_clusters(self, distances: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest centroid."""
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                         centroids: np.ndarray) -> float:
        """Compute WCSS = Σ_k Σ_{x ∈ C_k} ‖x − μ_k‖²"""
        return float(sum(
            np.sum((X[labels == k] - centroids[k]) ** 2)
            for k in range(self.n_clusters)
        ))

    def _run_single(self, X: np.ndarray):
        """
        Run one full Mini-Batch K-Means trial.

        At each iteration:
          1. Draw a random mini-batch B ⊂ X.
          2. Assign each x ∈ B to the nearest centroid.
          3. Update centroids with the online rule:
               n_k += 1;  η = 1/n_k;  μ_k ← (1−η)·μ_k + η·x

        Returns centroids, labels (full dataset), inertia, n_iter.
        """
        rng = np.random.RandomState(self.random_state)
        centroids = self._init_centroids(X)
        # Per-cluster running counts for the learning rate
        counts = np.zeros(self.n_clusters, dtype=np.float64)

        for iteration in range(1, self.max_iter + 1):
            # ── 1. Sample mini-batch ──────────────────────────────────
            batch_idx = rng.choice(X.shape[0],
                                   size=min(self.batch_size, X.shape[0]),
                                   replace=False)
            X_batch = X[batch_idx]

            # ── 2. Assign mini-batch points ───────────────────────────
            distances = self._compute_distances(X_batch, centroids)
            batch_labels = self._assign_clusters(distances)

            # ── 3. Online centroid update ─────────────────────────────
            old_centroids = centroids.copy()
            for i, k in enumerate(batch_labels):
                counts[k] += 1
                eta = 1.0 / counts[k]           # decreasing learning rate
                centroids[k] = (1 - eta) * centroids[k] + eta * X_batch[i]

            # ── 4. Convergence check ──────────────────────────────────
            shift = np.linalg.norm(centroids - old_centroids)
            if shift < self.tol:
                break

        # Final full-dataset assignment for reporting
        full_distances = self._compute_distances(X, centroids)
        labels = self._assign_clusters(full_distances)
        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def train(self, X: np.ndarray = None):
        """
        Fit Mini-Batch K-Means to the data.

        Runs n_init independent trials (each with fresh centroid initialisation)
        and keeps the solution with the lowest inertia.
        """
        if X is None:
            X = self.X_scaled

        best_inertia   = np.inf
        best_centroids = None
        best_labels    = None
        best_n_iter    = None

        print(f"\nTraining Mini-Batch K-Means (k={self.n_clusters}, "
              f"batch_size={self.batch_size}, n_init={self.n_init}) ...")

        for run in range(self.n_init):
            centroids, labels, inertia, n_iter = self._run_single(X)
            print(f"  Run {run + 1:2d}/{self.n_init} | "
                  f"Inertia: {inertia:,.2f} | Iters: {n_iter}")

            if inertia < best_inertia:
                best_inertia   = inertia
                best_centroids = centroids
                best_labels    = labels
                best_n_iter    = n_iter

        self.centroids = best_centroids
        self.labels_   = best_labels
        self.inertia_  = best_inertia
        self.n_iter_   = best_n_iter

        print(f"\nBest inertia: {self.inertia_:,.2f} "
              f"(converged in {self.n_iter_} iter)")
        return self

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """Assign new data points to the nearest centroid."""
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
            f"{self.config['saved_models']['names']['Mini-Batch K-Means Clustering']}"
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
            f"{self.config['saved_models']['names']['Mini-Batch K-Means Clustering']}"
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
        """Save clustering metrics to a JSON file."""
        import json as _json, os

        report_dir = self.config['plots']['dir']
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/minibatch_kmeans_report.json"

        with open(report_path, 'w') as f:
            _json.dump(metrics, f, indent=4)

        print("\n--- Mini-Batch K-Means Report ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"Report saved → {report_path}")

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def generate_plots(self, X_scaled: np.ndarray = None):
        """
        Generate and save all diagnostic plots:

        1. Elbow Method  — Inertia vs k (yellowbrick KElbowVisualizer).
        2. Silhouette    — Per-cluster silhouette analysis.
        3. PCA Scatter   — 2-D projection coloured by cluster label.
        4. Distribution  — Bar chart of cluster sizes.

        All files are saved with a '_minibatch' suffix to avoid collisions
        with the standard K-Means plots.
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
        mb_proxy = _SklearnMBKMeans(init=self.init_method,
                                    n_init=self.n_init,
                                    random_state=self.random_state,
                                    batch_size=self.batch_size)
        visualizer = KElbowVisualizer(mb_proxy, k=(2, 10), ax=ax)
        visualizer.fit(X_scaled)
        visualizer.finalize()
        ax.set_title('Elbow Method — Mini-Batch K-Means', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{names['elbow_method']}_minibatch.png", dpi=150)
        plt.close()

        # ── 2. Silhouette Analysis ───────────────────────────────────────
        print("Plotting silhouette analysis ...")
        fig, ax = plt.subplots(figsize=(8, 6))
        mb_proxy2 = _SklearnMBKMeans(n_clusters=self.n_clusters,
                                     init=self.init_method,
                                     n_init=self.n_init,
                                     random_state=self.random_state,
                                     batch_size=self.batch_size)
        sil_viz = SilhouetteVisualizer(mb_proxy2, colors='yellowbrick', ax=ax)
        sil_viz.fit(X_scaled)
        sil_viz.finalize()
        ax.set_title(f'Silhouette Analysis — Mini-Batch (k={self.n_clusters})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{names['silhouette_scores']}_minibatch.png", dpi=150)
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

        centroids_2d = pca.transform(self.centroids)
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   c='black', marker='X', s=180, zorder=5, label='Centroids')

        ax.set_title(f'Mini-Batch K-Means — PCA Projection (k={self.n_clusters})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{names['cluster_centers']}_minibatch.png", dpi=150)
        plt.close()

        # ── 4. Cluster Size Distribution ─────────────────────────────────
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
        ax.set_title(f'Cluster Distribution — Mini-Batch (k={self.n_clusters})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Points')
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{names['cluster_distribution']}_minibatch.png", dpi=150)
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

    config = json.load(open('ml_fundamentals/chapter3/configs/clustering.json'))

    os.makedirs(config['saved_models']['dir'], exist_ok=True)
    os.makedirs(config['plots']['dir'],        exist_ok=True)

    mb = MiniBatchKMeansClustering(config)
    mb.load_data(dataset_index=0)   # glass dataset
    mb.preprocess_data()
    mb.train()

    labels   = mb.labels_
    X_scaled = mb.X_scaled

    metrics = {
        "n_clusters":        mb.n_clusters,
        "batch_size":        mb.batch_size,
        "inertia":           round(mb.inertia_, 4),
        "silhouette_score":  round(silhouette_score(X_scaled, labels), 4),
        "davies_bouldin":    round(davies_bouldin_score(X_scaled, labels), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 4),
        "n_iterations":      mb.n_iter_,
    }

    mb.generate_report(metrics)
    mb.save_model()
    mb.generate_plots()
