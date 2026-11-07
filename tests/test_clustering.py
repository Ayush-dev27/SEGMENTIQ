"""
Unit tests for customer ML clustering module (src/analytics/clustering.py).
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.analytics.clustering import (
    preprocess_rfm,
    evaluate_clusters,
    fit_kmeans_clusters,
    get_cluster_summary,
    DEFAULT_FEATURES,
)


class TestClustering(unittest.TestCase):
    """Test suite for unsupervised RFM customer clustering."""

    def setUp(self):
        """Create a synthetic dataset with distinct behavioral clusters."""
        np.random.seed(42)
        n_samples = 60

        # Cluster 1: High recency (inactive), low frequency, low monetary
        c1 = {
            "CustomerID": list(range(100, 120)),
            "Recency": np.random.uniform(200, 350, 20),
            "Frequency": np.random.uniform(1, 2, 20),
            "Monetary": np.random.uniform(10, 100, 20),
        }
        # Cluster 2: Low recency (active), high frequency, high monetary
        c2 = {
            "CustomerID": list(range(120, 140)),
            "Recency": np.random.uniform(1, 30, 20),
            "Frequency": np.random.uniform(15, 30, 20),
            "Monetary": np.random.uniform(3000, 10000, 20),
        }
        # Cluster 3: Moderate recency, moderate frequency, moderate spend
        c3 = {
            "CustomerID": list(range(140, 160)),
            "Recency": np.random.uniform(40, 100, 20),
            "Frequency": np.random.uniform(4, 8, 20),
            "Monetary": np.random.uniform(400, 1500, 20),
        }

        dfs = [pd.DataFrame(c1), pd.DataFrame(c2), pd.DataFrame(c3)]
        self.sample_df = pd.concat(dfs, ignore_index=True)

    def test_preprocess_rfm_shapes_and_scaling(self):
        """Test 1: Check transformed data shape, zero mean (~0), and unit variance (~1)."""
        scaled_data, scaler = preprocess_rfm(self.sample_df)

        self.assertIsInstance(scaled_data, np.ndarray)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertEqual(scaled_data.shape, (len(self.sample_df), len(DEFAULT_FEATURES)))

        # Mean along each feature column should be close to 0 (tolerance 1e-6)
        means = np.mean(scaled_data, axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-6)

        # Standard deviation along each feature column should be close to 1 (tolerance 1e-6)
        stds = np.std(scaled_data, axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-6)

    def test_evaluate_clusters(self):
        """Test 2: Ensure returned dict has correct keys, monotonic non-increasing inertias, and valid silhouette score bounds."""
        scaled_data, _ = preprocess_rfm(self.sample_df)
        k_range = range(2, 6)
        results = evaluate_clusters(scaled_data, k_range=k_range, random_state=42)

        self.assertIn("k_values", results)
        self.assertIn("inertias", results)
        self.assertIn("silhouette_scores", results)

        self.assertEqual(results["k_values"], list(k_range))
        self.assertEqual(len(results["inertias"]), len(k_range))
        self.assertEqual(len(results["silhouette_scores"]), len(k_range))

        # Inertia strictly or monotonically decreases as k increases
        inertias = results["inertias"]
        for i in range(len(inertias) - 1):
            self.assertGreaterEqual(inertias[i], inertias[i + 1])

        # Silhouette score must lie within [-1, 1]
        for score in results["silhouette_scores"]:
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_fit_kmeans_clusters(self):
        """Test 3: Verify Cluster column presence, correct number of unique labels, and reproducibility."""
        n_clusters = 3
        df_clustered, model, scaler = fit_kmeans_clusters(
            self.sample_df, n_clusters=n_clusters, random_state=42
        )

        self.assertIn("Cluster", df_clustered.columns)
        self.assertIsInstance(model, KMeans)
        self.assertIsInstance(scaler, StandardScaler)

        # Check unique labels: exactly 0, 1, 2
        unique_labels = sorted(df_clustered["Cluster"].unique())
        self.assertEqual(unique_labels, list(range(n_clusters)))

        # Verify reproducibility with fixed random_state
        df_clustered_repeat, _, _ = fit_kmeans_clusters(
            self.sample_df, n_clusters=n_clusters, random_state=42
        )
        pd.testing.assert_series_equal(
            df_clustered["Cluster"], df_clustered_repeat["Cluster"]
        )

    def test_immutability(self):
        """Test 4: Input DataFrame remains completely unchanged after clustering."""
        original_copy = self.sample_df.copy()
        original_cols = list(self.sample_df.columns)

        df_clustered, _, _ = fit_kmeans_clusters(self.sample_df, n_clusters=3)

        # Original DataFrame unchanged
        self.assertNotIn("Cluster", self.sample_df.columns)
        self.assertListEqual(list(self.sample_df.columns), original_cols)
        pd.testing.assert_frame_equal(self.sample_df, original_copy)

        # Result is a new object
        self.assertIsNot(df_clustered, self.sample_df)

    def test_get_cluster_summary(self):
        """Test 5: Check output schema (Count, Percentage, mean/median metrics)."""
        df_clustered, _, _ = fit_kmeans_clusters(self.sample_df, n_clusters=3)
        summary = get_cluster_summary(df_clustered)

        expected_columns = [
            "Cluster",
            "Count",
            "Percentage",
            "Recency_Mean",
            "Recency_Median",
            "Frequency_Mean",
            "Frequency_Median",
            "Monetary_Mean",
            "Monetary_Median",
        ]
        for col in expected_columns:
            self.assertIn(col, summary.columns)

        # Row count matches number of clusters
        self.assertEqual(len(summary), 3)

        # Sum of counts equals total samples
        self.assertEqual(summary["Count"].sum(), len(self.sample_df))

        # Sum of percentage share ~ 100%
        self.assertAlmostEqual(summary["Percentage"].sum(), 100.0, places=1)

    def test_insufficient_data_error(self):
        """Test 6: Verify ValueError raised when row count < n_clusters."""
        tiny_df = pd.DataFrame(
            {
                "CustomerID": [1, 2],
                "Recency": [10, 20],
                "Frequency": [1, 2],
                "Monetary": [50.0, 100.0],
            }
        )
        with self.assertRaises(ValueError):
            fit_kmeans_clusters(tiny_df, n_clusters=5)

    def test_missing_columns_error(self):
        """Test 7: Verify ValueError when RFM columns are absent."""
        invalid_df = pd.DataFrame({"CustomerID": [101, 102], "RandomCol": [1, 2]})

        with self.assertRaises(ValueError):
            preprocess_rfm(invalid_df)

        with self.assertRaises(ValueError):
            fit_kmeans_clusters(invalid_df)


if __name__ == "__main__":
    unittest.main()
