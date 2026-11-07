"""
RFM scoring module for SEGMENTIQ.

Responsible for converting customer-level RFM metrics (Recency, Frequency, Monetary)
into quantile-based discrete scores (1 to 5), handling duplicate bin edges robustly,
and computing compound score representations (RFMScore and RFMTOTAL).
"""

from typing import Sequence
import pandas as pd

# Required input columns for RFM scoring
REQUIRED_COLUMNS: tuple[str, ...] = (
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
)


def validate_scoring_input(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> None:
    """
    Validate that the input DataFrame contains all required columns for scoring.

    Parameters:
        df (pd.DataFrame): Input customer-level RFM DataFrame.
        required_columns (Sequence[str]): Required column names.
            Defaults to REQUIRED_COLUMNS.

    Raises:
        ValueError: If df is None or required columns are missing.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing required columns for RFM scoring: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )


def compute_quantile_score(
    series: pd.Series,
    ascending: bool = True,
    q: int = 5,
) -> pd.Series:
    """
    Calculate discrete quantile-based scores handling duplicated bin boundaries robustly.

    Parameters:
        series (pd.Series): Metric values to bin into quantiles.
        ascending (bool):
            - If True (Frequency, Monetary): Higher metric value gets higher score (1 to k).
            - If False (Recency): Lower metric value (more recent) gets higher score (k to 1).
        q (int): Number of quantile buckets. Defaults to 5.

    Returns:
        pd.Series: Integer scores indexed identically to input series.
    """
    if series.empty:
        return pd.Series(dtype="int64", index=series.index)

    if series.nunique() <= 1:
        return pd.Series(1, index=series.index, dtype="int64")

    # Use duplicates='drop' to prevent crash when quantile boundaries collapse
    bins, bin_edges = pd.qcut(
        series, q=q, duplicates="drop", labels=False, retbins=True
    )
    num_bins = len(bin_edges) - 1

    if num_bins <= 1:
        return pd.Series(1, index=series.index, dtype="int64")

    bins_series = pd.Series(bins, index=series.index).fillna(0).astype("int64")

    if ascending:
        # Lower bucket gets 1, highest bucket gets num_bins
        score = bins_series + 1
    else:
        # Lower bucket (most recent) gets num_bins, highest bucket gets 1
        score = num_bins - bins_series

    return score.astype("int64")


def score_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score customer RFM metrics into quantile-based buckets (1 to 5).

    Pipeline operations:
    1. Validate required input columns (CustomerID, Recency, Frequency, Monetary).
    2. Work on a non-destructive copy of the input.
    3. Score Recency inversely (lower days = higher score).
    4. Score Frequency positively (higher transactions = higher score).
    5. Score Monetary positively (higher spend = higher score).
    6. Construct RFMScore as compact string (e.g. '545').
    7. Compute RFMTOTAL as sum of the individual scores.

    Note:
        Strictly handles scoring. Does NOT perform clustering, customer
        segmentation, recommendations, or visualization.

    Parameters:
        df (pd.DataFrame): Customer-level RFM metrics DataFrame.

    Returns:
        pd.DataFrame: New DataFrame containing original metrics along with:
            - RecencyScore (int64)
            - FrequencyScore (int64)
            - MonetaryScore (int64)
            - RFMScore (str)
            - RFMTOTAL (int64)

    Raises:
        ValueError: If input DataFrame is None or missing required columns.
    """
    validate_scoring_input(df)

    if df.empty:
        result = df.copy()
        result["RecencyScore"] = pd.Series(dtype="int64")
        result["FrequencyScore"] = pd.Series(dtype="int64")
        result["MonetaryScore"] = pd.Series(dtype="int64")
        result["RFMScore"] = pd.Series(dtype="str")
        result["RFMTOTAL"] = pd.Series(dtype="int64")
        return result

    result = df.copy()

    # Recency: lower value = more recent = higher score (descending)
    result["RecencyScore"] = compute_quantile_score(
        result["Recency"], ascending=False, q=5
    )

    # Frequency: higher value = more frequent = higher score (ascending)
    result["FrequencyScore"] = compute_quantile_score(
        result["Frequency"], ascending=True, q=5
    )

    # Monetary: higher value = higher spend = higher score (ascending)
    result["MonetaryScore"] = compute_quantile_score(
        result["Monetary"], ascending=True, q=5
    )

    # Compound string score (e.g., '545')
    result["RFMScore"] = (
        result["RecencyScore"].astype(str)
        + result["FrequencyScore"].astype(str)
        + result["MonetaryScore"].astype(str)
    )

    # Total integer score
    result["RFMTOTAL"] = (
        result["RecencyScore"]
        + result["FrequencyScore"]
        + result["MonetaryScore"]
    ).astype("int64")

    return result


__all__ = [
    "score_rfm",
    "compute_quantile_score",
    "validate_scoring_input",
    "REQUIRED_COLUMNS",
]
