"""
API router for SEGMENTIQ customer analytics and retention engine.

Exposes RESTful endpoints for health checks, high-level business KPI summaries,
paginated and filtered customer tables, and single customer detail views.
"""

import math
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/v1", tags=["Segmentation"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customers_rfm.csv"

_cached_df: Optional[pd.DataFrame] = None


def get_processed_data(
    file_path: Optional[Union[str, Path]] = None,
    reload: bool = False,
) -> pd.DataFrame:
    """
    Retrieve processed customer RFM data with in-memory caching.

    Parameters:
        file_path (str | Path | None): Optional custom path to customers_rfm.csv.
        reload (bool): If True, reloads the file from disk, refreshing the cache.

    Returns:
        pd.DataFrame: Loaded customer dataset or empty DataFrame if not found.
    """
    global _cached_df
    if _cached_df is not None and not reload:
        return _cached_df

    target_path = Path(file_path) if file_path else DEFAULT_DATA_PATH
    if not target_path.exists() or not target_path.is_file():
        return pd.DataFrame()

    try:
        df = pd.read_csv(target_path)
        _cached_df = df
        return _cached_df
    except Exception:
        return pd.DataFrame()


def set_processed_data(df: pd.DataFrame) -> None:
    """Manually override in-memory cached dataset (useful for unit testing)."""
    global _cached_df
    _cached_df = df


@router.get("/health", summary="Health Check")
def health_check() -> dict[str, Any]:
    """Check API health and confirm if the processed customer dataset is loaded."""
    df = get_processed_data()
    loaded = not df.empty
    return {
        "status": "ok",
        "dataset_loaded": loaded,
        "total_customers": int(len(df)) if loaded else 0,
    }


@router.get("/summary", summary="Executive KPI Summary")
def get_summary() -> dict[str, Any]:
    """
    Compute high-level business KPI metrics and segment distributions
    for dashboard metric cards and analytics visualizations.
    """
    df = get_processed_data()
    if df.empty:
        return {
            "total_customers": 0,
            "total_revenue": 0.0,
            "avg_recency": 0.0,
            "avg_frequency": 0.0,
            "avg_monetary": 0.0,
            "segment_distribution": [],
            "cluster_distribution": [],
            "priority_distribution": [],
        }

    total_customers = int(len(df))
    total_revenue = float(round(df["Monetary"].sum(), 2))
    avg_recency = float(round(df["Recency"].mean(), 2))
    avg_frequency = float(round(df["Frequency"].mean(), 2))
    avg_monetary = float(round(df["Monetary"].mean(), 2))

    # 1. Segment Distribution
    seg_grouped = df.groupby("Segment", observed=False)
    seg_counts = seg_grouped.size()
    segment_distribution = []
    for seg_name, count in seg_counts.items():
        sub = df[df["Segment"] == seg_name]
        segment_distribution.append(
            {
                "segment": str(seg_name),
                "count": int(count),
                "percentage": float(round((count / total_customers) * 100, 2)),
                "mean_recency": float(round(sub["Recency"].mean(), 2)),
                "mean_frequency": float(round(sub["Frequency"].mean(), 2)),
                "mean_monetary": float(round(sub["Monetary"].mean(), 2)),
            }
        )
    segment_distribution.sort(key=lambda x: x["count"], reverse=True)

    # 2. Cluster Distribution
    cluster_distribution = []
    if "Cluster" in df.columns:
        clus_grouped = df.groupby("Cluster", observed=False)
        for clus_id, count in clus_grouped.size().items():
            sub = df[df["Cluster"] == clus_id]
            cluster_distribution.append(
                {
                    "cluster": int(clus_id),
                    "count": int(count),
                    "percentage": float(round((count / total_customers) * 100, 2)),
                    "mean_monetary": float(round(sub["Monetary"].mean(), 2)),
                }
            )
        cluster_distribution.sort(key=lambda x: x["cluster"])

    # 3. Priority Distribution
    priority_distribution = []
    if "Action_Priority" in df.columns:
        prio_counts = df["Action_Priority"].value_counts()
        prio_order = {"Urgent": 0, "High": 1, "Medium": 2, "Low": 3}
        for prio_name, count in prio_counts.items():
            priority_distribution.append(
                {"priority": str(prio_name), "count": int(count)}
            )
        priority_distribution.sort(key=lambda x: prio_order.get(x["priority"], 99))

    return {
        "total_customers": total_customers,
        "total_revenue": total_revenue,
        "avg_recency": avg_recency,
        "avg_frequency": avg_frequency,
        "avg_monetary": avg_monetary,
        "segment_distribution": segment_distribution,
        "cluster_distribution": cluster_distribution,
        "priority_distribution": priority_distribution,
    }


@router.get("/customers", summary="Paginated Customers List")
def get_customers(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    segment: Optional[str] = Query(None, description="Filter by segment name (case-insensitive)"),
    cluster: Optional[int] = Query(None, description="Filter by K-Means cluster ID"),
    priority: Optional[str] = Query(None, description="Filter by Action_Priority (case-insensitive)"),
    sort_by: str = Query("CustomerID", description="Column to sort by"),
    order: str = Query("asc", description="Sort order: 'asc' or 'desc'"),
) -> dict[str, Any]:
    """
    Retrieve paginated customer records with filtering and sorting support for data tables.
    """
    df = get_processed_data()
    if df.empty:
        return {
            "total": 0,
            "page": page,
            "limit": limit,
            "total_pages": 0,
            "customers": [],
        }

    filtered = df.copy()

    # Segment filter
    if segment:
        filtered = filtered[
            filtered["Segment"].astype(str).str.strip().str.lower()
            == segment.strip().lower()
        ]

    # Cluster filter
    if cluster is not None and "Cluster" in filtered.columns:
        filtered = filtered[filtered["Cluster"] == cluster]

    # Action priority filter
    if priority and "Action_Priority" in filtered.columns:
        filtered = filtered[
            filtered["Action_Priority"].astype(str).str.strip().str.lower()
            == priority.strip().lower()
        ]

    # Sorting
    allowed_sort_cols = {
        "customerid": "CustomerID",
        "monetary": "Monetary",
        "recency": "Recency",
        "frequency": "Frequency",
        "rfmtotal": "RFMTOTAL",
    }
    canonical_sort_col = allowed_sort_cols.get(sort_by.lower(), "CustomerID")
    if canonical_sort_col in filtered.columns:
        ascending = order.lower() != "desc"
        filtered = filtered.sort_values(by=canonical_sort_col, ascending=ascending)

    total_records = len(filtered)
    total_pages = math.ceil(total_records / limit) if total_records > 0 else 0

    # Paginate slice
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    page_records = filtered.iloc[start_idx:end_idx]

    customers_list = page_records.replace({np.nan: None}).to_dict(orient="records")

    return {
        "total": total_records,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "customers": customers_list,
    }


@router.get("/customers/{customer_id}", summary="Single Customer Detail")
def get_customer(customer_id: int) -> dict[str, Any]:
    """
    Fetch a single customer's full behavioral metrics, scores, segment,
    cluster, and personalized retention action strategy.
    """
    df = get_processed_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    match = df[df["CustomerID"] == customer_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    record = match.iloc[0].replace({np.nan: None}).to_dict()
    return record


__all__ = [
    "router",
    "get_processed_data",
    "set_processed_data",
    "DEFAULT_DATA_PATH",
]
