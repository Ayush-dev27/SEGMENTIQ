"""
Data cleaning and preprocessing module for SEGMENTIQ.

Responsible for sanitizing raw Online Retail transaction records, filtering invalid
records (cancellations, non-positive quantities/prices, missing customers, duplicates),
handling missing descriptions without fabricating data, and enforcing strict data types.
"""

from typing import Sequence
import numpy as np
import pandas as pd

# Expected columns required for the cleaning pipeline
REQUIRED_COLUMNS: tuple[str, ...] = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)


def remove_missing_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where CustomerID is missing, null, or empty.

    Parameters:
        df (pd.DataFrame): DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only records with a valid CustomerID.
    """
    mask = df["CustomerID"].notna()
    filtered = df[mask].copy()
    # In case CustomerID was read or formatted as string 'nan', 'none', or whitespace
    str_mask = ~filtered["CustomerID"].astype(str).str.strip().str.lower().isin(
        ["nan", "none", "null", ""]
    )
    return filtered[str_mask].copy()


def remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancelled transactions where InvoiceNo starts with 'C' (case-insensitive).

    Parameters:
        df (pd.DataFrame): DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only active (non-cancelled) transactions.
    """
    is_cancelled = (
        df["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C")
    )
    return df[~is_cancelled].copy()


def remove_invalid_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove transactions where Quantity is less than or equal to 0 or non-numeric.

    Parameters:
        df (pd.DataFrame): DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only positive quantities.
    """
    qty_numeric = pd.to_numeric(df["Quantity"], errors="coerce")
    valid_mask = qty_numeric.notna() & (qty_numeric > 0)
    return df[valid_mask].copy()


def remove_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove transactions where UnitPrice is less than or equal to 0 or non-numeric.

    Parameters:
        df (pd.DataFrame): DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only positive unit prices.
    """
    price_numeric = pd.to_numeric(df["UnitPrice"], errors="coerce")
    valid_mask = price_numeric.notna() & (price_numeric > 0)
    return df[valid_mask].copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate records across all columns.

    Parameters:
        df (pd.DataFrame): DataFrame to deduplicate.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    return df.drop_duplicates().copy()


def handle_descriptions(
    df: pd.DataFrame, fill_value: str = "Unknown"
) -> pd.DataFrame:
    """
    Handle missing or blank product descriptions appropriately without inventing info.

    Parameters:
        df (pd.DataFrame): DataFrame containing Description column.
        fill_value (str): Standardized neutral placeholder for missing values. Defaults to 'Unknown'.

    Returns:
        pd.DataFrame: DataFrame with normalized description strings.
    """
    cleaned = df.copy()
    desc = cleaned["Description"].fillna(fill_value).astype(str).str.strip()
    cleaned["Description"] = desc.replace("", fill_value)
    return cleaned


def enforce_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to appropriate production datatypes:
    - CustomerID: int64
    - InvoiceDate: datetime64[ns]
    - Quantity: int64
    - UnitPrice: float64

    Parameters:
        df (pd.DataFrame): Validated DataFrame.

    Returns:
        pd.DataFrame: DataFrame with enforced column datatypes.
    """
    cleaned = df.copy()
    cleaned["CustomerID"] = pd.to_numeric(cleaned["CustomerID"]).astype("int64")
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"]).astype("int64")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"]).astype("float64")
    return cleaned


def clean_data(
    df: pd.DataFrame,
    description_fill: str = "Unknown",
    expected_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """
    Execute the complete cleaning and validation pipeline for SEGMENTIQ raw data.

    Pipeline operations:
    1. Validate presence of required schema columns.
    2. Remove rows with missing/null CustomerID.
    3. Remove cancelled transactions (InvoiceNo starting with 'C').
    4. Remove rows where Quantity <= 0.
    5. Remove rows where UnitPrice <= 0.
    6. Remove exact duplicate rows.
    7. Normalize Description without inventing product information (neutral placeholder).
    8. Enforce target production data types (CustomerID: int64, InvoiceDate: datetime,
       Quantity: int64, UnitPrice: float64).
    9. Reset the index.

    Note:
        This module strictly performs data sanitization. It does NOT compute RFM
        metrics, customer segments, clusters, or recommendation logic.

    Parameters:
        df (pd.DataFrame): Raw transaction DataFrame.
        description_fill (str): Neutral placeholder for missing descriptions.
            Defaults to 'Unknown'.
        expected_columns (Sequence[str]): Required column names.
            Defaults to REQUIRED_COLUMNS.

    Returns:
        pd.DataFrame: Sanitized, strongly-typed DataFrame with reset index.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing_cols}. "
            f"Found: {list(df.columns)}"
        )

    if df.empty:
        return df.copy().reset_index(drop=True)

    cleaned = df.copy()
    cleaned = remove_missing_customers(cleaned)
    cleaned = remove_cancellations(cleaned)
    cleaned = remove_invalid_quantities(cleaned)
    cleaned = remove_invalid_prices(cleaned)
    cleaned = remove_duplicates(cleaned)

    if cleaned.empty:
        return cleaned.reset_index(drop=True)

    cleaned = handle_descriptions(cleaned, fill_value=description_fill)
    cleaned = enforce_datatypes(cleaned)
    return cleaned.reset_index(drop=True)


__all__ = [
    "clean_data",
    "remove_missing_customers",
    "remove_cancellations",
    "remove_invalid_quantities",
    "remove_invalid_prices",
    "remove_duplicates",
    "handle_descriptions",
    "enforce_datatypes",
    "REQUIRED_COLUMNS",
]
