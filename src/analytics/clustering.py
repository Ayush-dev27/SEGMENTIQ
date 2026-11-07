"""
Unsupervised machine learning customer clustering module for SEGMENTIQ.

Responsible for preprocessing RFM feature distributions, evaluating optimal
cluster counts (Elbow inertia & Silhouette analysis), fitting K-Means clustering
models, and generating cluster profile summaries.
"""

from typing import Sequence, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["Recency", "Frequency", "Monetary"]


def preprocess_rfm(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Preprocess RFM metrics using log1p transformation and standard scaling.

    RFM metrics are typically heavily right-skewed and positive. Log transformation
    (log1p) compresses extreme positive values while safely handling zeroes,
    and StandardScaler standardizes features to zero mean and unit variance.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing RFM metrics.
        feature_cols (Sequence[str]): Feature column names to transform and scale.
            Defaults to ['Recency', 'Frequency', 'Monetary'].

    Returns:
        tuple[np.ndarray, StandardScaler]:
            - Scaled feature matrix of shape (n_samples, n_features).
            - Fitted StandardScaler instance.

    Raises:
        ValueError: If df is None, empty, missing required columns, or contains negative values.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing required feature columns: {missing_cols}. "
            f"Found: {list(df.columns)}"
        )

    feature_data = df[list(feature_cols)].copy()

    # Verify numeric and non-negative
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(feature_data[col]):
            raise ValueError(f"Feature column '{col}' must contain numeric data.")
        if (feature_data[col] < 0).any():
            raise ValueError(
                f"Feature column '{col}' contains negative values, which cannot be log-transformed."
            )

    # 1. Log-transform with log1p: log(1 + x)
    log_transformed = np.log1p(feature_data.values.astype(float))

    # 2. Standardize to zero mean and unit variance
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(log_transformed)

    return scaled_data, scaler


def evaluate_clusters(
    scaled_data: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42,
) -> dict[str, list]:
    """
    Evaluate K-Means clustering quality across a range of k values.

    Computes inertia (within-cluster sum of squares) for the Elbow Method
    and Silhouette Scores to determine cluster cohesion and separation.

    Parameters:
        scaled_data (np.ndarray): Scaled 2D feature matrix of shape (n_samples, n_features).
        k_range (range): Range of k values to evaluate. Defaults to range(2, 9).
        random_state (int): Random state for reproducibility. Defaults to 42.

    Returns:
        dict[str, list]: Dictionary containing:
            - 'k_values': list of evaluated k integers.
            - 'inertias': list of float inertia values.
            - 'silhouette_scores': list of float silhouette scores [-1, 1].

    Raises:
        ValueError: If scaled_data is None, empty, or has fewer samples than the minimum k.
    """
    if scaled_data is None:
        raise ValueError("Scaled data cannot be None.")

    if len(scaled_data) < 2:
        raise ValueError(
            f"Scaled data must have at least 2 samples, found {len(scaled_data)}."
        )

    k_list = list(k_range)
    if not k_list:
        raise ValueError("k_range cannot be empty.")

    if min(k_list) < 2:
        raise ValueError("Minimum k must be at least 2 for silhouette evaluation.")

    if max(k_list) >= len(scaled_data):
        raise ValueError(
            f"Maximum k ({max(k_list)}) must be strictly less than the number of samples ({len(scaled_data)})."
        )

    inertias: list[float] = []
    silhouette_scores: list[float] = []

    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        inertias.append(float(kmeans.inertia_))
        score = float(silhouette_score(scaled_data, labels))
        silhouette_scores.append(score)

    return {
        "k_values": k_list,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
    }


def fit_kmeans_clusters(
    df: pd.DataFrame,
    n_clusters: int = 4,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Fit K-Means clustering to RFM features and append cluster assignments.

    Non-destructive: operates on a copy of `df` and adds a new integer 'Cluster' column.

    Parameters:
        df (pd.DataFrame): Input customer-level RFM DataFrame.
        n_clusters (int): Number of clusters to form. Defaults to 4.
        feature_cols (Sequence[str]): Columns to use for clustering.
            Defaults to ['Recency', 'Frequency', 'Monetary'].
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, KMeans, StandardScaler]:
            - Copy of df with new 'Cluster' column (values 0 to n_clusters-1).
            - Fitted KMeans model instance.
            - Fitted StandardScaler instance.

    Raises:
        ValueError: If df is None, missing required columns, or len(df) < n_clusters.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if len(df) < n_clusters:
        raise ValueError(
            f"Input DataFrame has {len(df)} rows, which is fewer than n_clusters={n_clusters}."
        )

    scaled_data, scaler = preprocess_rfm(df, feature_cols=feature_cols)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)

    result_df = df.copy()
    result_df["Cluster"] = cluster_labels.astype(int)

    return result_df, kmeans, scaler


def get_cluster_summary(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """
    Compute cluster profile summary statistics.

    Parameters:
        df (pd.DataFrame): Clustered DataFrame containing `cluster_col`.
        cluster_col (str): Column name containing cluster labels. Defaults to 'Cluster'.
        feature_cols (Sequence[str]): Feature columns to profile.
            Defaults to ['Recency', 'Frequency', 'Monetary'].

    Returns:
        pd.DataFrame: Summary table sorted by cluster index containing:
            - Cluster (int)
            - Count (int)
            - Percentage (float)
            - Mean and Median for each available feature column.

    Raises:
        ValueError: If df is None or cluster_col is missing.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if cluster_col not in df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in DataFrame.")

    if df.empty:
        cols = [cluster_col, "Count", "Percentage"]
        for f in feature_cols:
            cols.extend([f"{f}_Mean", f"{f}_Median"])
        return pd.DataFrame(columns=cols)

    total_customers = len(df)
    grouped = df.groupby(cluster_col, observed=False)
    counts = grouped.size()
    percentages = (counts / total_customers * 100).round(2)

    summary_data = {
        cluster_col: counts.index.values,
        "Count": counts.values,
        "Percentage": percentages.values,
    }

    for col in feature_cols:
        if col in df.columns:
            mean_vals = grouped[col].mean().round(2).values
            med_vals = grouped[col].median().round(2).values
            summary_data[f"{col}_Mean"] = mean_vals
            summary_data[f"{col}_Median"] = med_vals
            # Also provide Mean/Median prefixed aliases
            summary_data[f"Mean{col}"] = mean_vals
            summary_data[f"Median{col}"] = med_vals

    summary_df = pd.DataFrame(summary_data).sort_values(cluster_col).reset_index(drop=True)
    return summary_df


__all__ = [
    "preprocess_rfm",
    "evaluate_clusters",
    "fit_kmeans_clusters",
    "get_cluster_summary",
    "DEFAULT_FEATURES",
]
