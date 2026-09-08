"""
RFM (Recency, Frequency, Monetary) metrics calculation module for SEGMENTIQ.

Responsible for aggregating transaction-level purchase history into customer-level
behavioral metrics used for downstream segmentation and lifecycle modeling.
"""

from datetime import datetime
from typing import Optional, Sequence, Union
import pandas as pd

# Expected columns required for RFM calculation
REQUIRED_COLUMNS: tuple[str, ...] = (
    "InvoiceNo",
    "InvoiceDate",
    "Quantity",
    "UnitPrice",
    "CustomerID",
)


def validate_rfm_input(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> None:
    """
    Validate that the input DataFrame contains all required columns for RFM calculation.

    Parameters:
        df (pd.DataFrame): Input transaction-level DataFrame.
        required_columns (Sequence[str]): Required column names.
            Defaults to REQUIRED_COLUMNS.

    Raises:
        ValueError: If df is None or any required columns are missing.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing required columns for RFM calculation: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )


def determine_reference_date(
    df: pd.DataFrame,
    reference_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
) -> pd.Timestamp:
    """
    Resolve the reference date used for Recency calculation.

    If not supplied, defaults to exactly one day after the latest InvoiceDate
    present in the dataset.

    Parameters:
        df (pd.DataFrame): Transaction DataFrame containing InvoiceDate.
        reference_date (str | datetime | pd.Timestamp | None): Optional reference date.

    Returns:
        pd.Timestamp: Resolved reference timestamp.
    """
    if reference_date is not None:
        return pd.to_datetime(reference_date)

    return pd.to_datetime(df["InvoiceDate"].max()) + pd.Timedelta(days=1)


def calculate_rfm(
    df: pd.DataFrame,
    reference_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Calculate customer-level Recency, Frequency, and Monetary (RFM) metrics.

    Parameters:
        df (pd.DataFrame):
            Cleaned transaction-level DataFrame containing at least:
            InvoiceNo, InvoiceDate, Quantity, UnitPrice, CustomerID.
        reference_date (str | datetime | pd.Timestamp | None):
            Snapshot reference date for Recency calculation. If None, defaults to
            one day after the latest InvoiceDate in `df`.

    Returns:
        pd.DataFrame:
            A customer-level DataFrame containing exactly:
            - CustomerID (int64): Unique customer identifier.
            - Recency (int64): Number of days between reference date and latest InvoiceDate.
            - Frequency (int64): Number of unique InvoiceNo values per customer.
            - Monetary (float64): Total spend (sum of Quantity * UnitPrice) per customer.

    Raises:
        ValueError:
            If input DataFrame is None or required columns are missing.
    """
    validate_rfm_input(df)

    if df.empty:
        return pd.DataFrame(
            columns=["CustomerID", "Recency", "Frequency", "Monetary"]
        )

    calc_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(calc_df["InvoiceDate"]):
        calc_df["InvoiceDate"] = pd.to_datetime(calc_df["InvoiceDate"])

    ref_dt = determine_reference_date(calc_df, reference_date=reference_date)

    # Calculate line-item transaction spend
    calc_df["TotalAmount"] = calc_df["Quantity"] * calc_df["UnitPrice"]

    # Aggregate by customer
    grouped = calc_df.groupby("CustomerID", as_index=False).agg(
        LatestInvoiceDate=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    )

    # Compute Recency in integer days
    grouped["Recency"] = (ref_dt - grouped["LatestInvoiceDate"]).dt.days

    # Filter and arrange final schema
    rfm = grouped[["CustomerID", "Recency", "Frequency", "Monetary"]].copy()

    # Enforce explicit production types
    rfm["CustomerID"] = pd.to_numeric(rfm["CustomerID"]).astype("int64")
    rfm["Recency"] = rfm["Recency"].astype("int64")
    rfm["Frequency"] = rfm["Frequency"].astype("int64")
    rfm["Monetary"] = rfm["Monetary"].astype("float64")

    return rfm


__all__ = [
    "calculate_rfm",
    "determine_reference_date",
    "validate_rfm_input",
    "REQUIRED_COLUMNS",
]
