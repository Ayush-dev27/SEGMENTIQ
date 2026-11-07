"""
Customer segmentation module for SEGMENTIQ.

Responsible for mapping customer RFM scores (Recency, Frequency, Monetary)
to business-interpretable customer lifecycle segments (e.g. Champions,
Loyal Customers, At Risk, Lost) using industry-standard RFM heuristics.
"""

from typing import Sequence, Union
import pandas as pd

# Standard segment names
SEGMENT_CHAMPIONS = "Champions"
SEGMENT_CANT_LOSE = "Can't Lose Them"
SEGMENT_LOYAL = "Loyal Customers"
SEGMENT_AT_RISK = "At Risk"
SEGMENT_NEW = "New Customers"
SEGMENT_PROMISING = "Promising"
SEGMENT_NEED_ATTENTION = "Customers Needing Attention"
SEGMENT_POTENTIAL_LOYAL = "Potential Loyalists"
SEGMENT_ABOUT_TO_SLEEP = "About to Sleep"
SEGMENT_LOST = "Lost"
SEGMENT_HIBERNATING = "Hibernating"
SEGMENT_OTHERS = "Others"

ALL_SEGMENTS: tuple[str, ...] = (
    SEGMENT_CHAMPIONS,
    SEGMENT_CANT_LOSE,
    SEGMENT_LOYAL,
    SEGMENT_AT_RISK,
    SEGMENT_NEW,
    SEGMENT_PROMISING,
    SEGMENT_NEED_ATTENTION,
    SEGMENT_POTENTIAL_LOYAL,
    SEGMENT_ABOUT_TO_SLEEP,
    SEGMENT_LOST,
    SEGMENT_HIBERNATING,
    SEGMENT_OTHERS,
)


def _classify_rfm_tuple(r: int, f: int, m: int, fallback: str = SEGMENT_OTHERS) -> str:
    """Classify a single (R, F, M) integer score tuple into a customer segment."""
    # 1. Champions: R: 4-5, F: 4-5, M: 4-5 (Bought recently, buy often, and spend the most)
    if r in (4, 5) and f in (4, 5) and m in (4, 5):
        return SEGMENT_CHAMPIONS

    # 2. Loyal Customers: R: 2-5, F: 3-5, M: 3-5 (Consistent spenders, responsive to promos)
    if r in (3, 4, 5) and f in (3, 4, 5) and m in (3, 4, 5):
        return SEGMENT_LOYAL

    # 3. Can't Lose Them: R: 1-2, F: 4-5, M: 4-5 (High spenders & frequent buyers who haven't returned)
    if r in (1, 2) and f in (4, 5) and m in (4, 5):
        return SEGMENT_CANT_LOSE

    # 4. Recent Customers / New Customers: R: 4-5, F: 1, M: 1-2 (Bought very recently, single purchase)
    if r in (4, 5) and f == 1 and m in (1, 2):
        return SEGMENT_NEW

    # 5. Promising: R: 3-4, F: 1, M: 1 (Recent shoppers, haven't spent much yet)
    if r in (3, 4) and f == 1 and m == 1:
        return SEGMENT_PROMISING

    # 6. Customers Needing Attention: R: 2-3, F: 2-3, M: 2-3 (Above average R, F & M values)
    if r in (2, 3) and f in (2, 3) and m in (2, 3):
        return SEGMENT_NEED_ATTENTION

    # 7. At Risk: R: 1-2, F: 2-5, M: 2-5 (Spent good money and bought often, but long time ago)
    if r in (1, 2) and f in (2, 3, 4, 5) and m in (2, 3, 4, 5):
        return SEGMENT_AT_RISK

    # 8. Potential Loyalists: R: 3-5, F: 1-3, M: 1-3 (Recent buyers with average frequency)
    if r in (3, 4, 5) and f in (1, 2, 3) and m in (1, 2, 3):
        return SEGMENT_POTENTIAL_LOYAL

    # 9. About to Sleep: R: 2-3, F: 1-2, M: 1-2 (Below average recency and frequency, will be lost)
    if r in (2, 3) and f in (1, 2) and m in (1, 2):
        return SEGMENT_ABOUT_TO_SLEEP

    # 10. Lost: R: 1, F: 1, M: 1 (Lowest scores across all dimensions)
    if r == 1 and f == 1 and m == 1:
        return SEGMENT_LOST

    # 11. Hibernating: R: 1-2, F: 1-2, M: 1-2 (Low spenders, low frequency, last purchase long ago)
    if r in (1, 2) and f in (1, 2) and m in (1, 2):
        return SEGMENT_HIBERNATING

    return fallback


def extract_rfm_score_series(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extract R, F, M integer series from df supporting multiple column naming conventions:
    - ('RecencyScore', 'FrequencyScore', 'MonetaryScore')
    - ('R_Score', 'F_Score', 'M_Score')
    - ('R', 'F', 'M')
    - or 3-digit string ('RFMScore', 'RFM_Score', 'RFM_Segment').

    Raises:
        ValueError: If df is None or required score columns cannot be found.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    # 1. Standard cleaner/scoring naming
    if {"RecencyScore", "FrequencyScore", "MonetaryScore"}.issubset(df.columns):
        return (
            pd.to_numeric(df["RecencyScore"], errors="coerce").fillna(1).astype(int),
            pd.to_numeric(df["FrequencyScore"], errors="coerce").fillna(1).astype(int),
            pd.to_numeric(df["MonetaryScore"], errors="coerce").fillna(1).astype(int),
        )

    # 2. R_Score / F_Score / M_Score
    if {"R_Score", "F_Score", "M_Score"}.issubset(df.columns):
        return (
            pd.to_numeric(df["R_Score"], errors="coerce").fillna(1).astype(int),
            pd.to_numeric(df["F_Score"], errors="coerce").fillna(1).astype(int),
            pd.to_numeric(df["M_Score"], errors="coerce").fillna(1).astype(int),
        )

    # 3. R / F / M
    if {"R", "F", "M"}.issubset(df.columns):
        return (
            pd.to_numeric(df["R"], errors="coerce").fillna(1).astype(int),
            pd.to_numeric(df["F"], errors="coerce").fillna(1).astype(int),
            pd.to_numeric(df["M"], errors="coerce").fillna(1).astype(int),
        )

    # 4. String format like '545', '111'
    for str_col in ("RFMScore", "RFM_Score", "RFM_Segment"):
        if str_col in df.columns:
            str_s = df[str_col].astype(str)
            # Ensure 3-character format
            if str_s.str.len().ge(3).all():
                r = pd.to_numeric(str_s.str[0], errors="coerce").fillna(1).astype(int)
                f = pd.to_numeric(str_s.str[1], errors="coerce").fillna(1).astype(int)
                m = pd.to_numeric(str_s.str[2], errors="coerce").fillna(1).astype(int)
                return r, f, m

    raise ValueError(
        "Input DataFrame is missing required RFM score columns. "
        "Expected one of: ('RecencyScore', 'FrequencyScore', 'MonetaryScore'), "
        "('R_Score', 'F_Score', 'M_Score'), ('R', 'F', 'M'), "
        "or string column ('RFMScore', 'RFM_Score', 'RFM_Segment'). "
        f"Found columns: {list(df.columns)}"
    )


def assign_segments(
    df: pd.DataFrame,
    fallback: str = SEGMENT_OTHERS,
) -> pd.DataFrame:
    """
    Assign business-interpretable customer segments to a DataFrame containing RFM scores.

    This function does NOT modify the input DataFrame in-place; it returns a new DataFrame copy
    with an added `Segment` column.

    Parameters:
        df (pd.DataFrame):
            DataFrame with score columns or RFMScore string.
        fallback (str):
            Default segment for any unassigned edge cases. Defaults to 'Others'.

    Returns:
        pd.DataFrame:
            A new DataFrame containing all original columns plus `Segment`.

    Raises:
        ValueError:
            If input DataFrame is None or missing required score columns.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    if df.empty:
        result = df.copy()
        result["Segment"] = pd.Series(dtype="object")
        return result

    r_series, f_series, m_series = extract_rfm_score_series(df)

    result = df.copy()
    segments = [
        _classify_rfm_tuple(int(r), int(f), int(m), fallback=fallback)
        for r, f, m in zip(r_series, f_series, m_series)
    ]
    result["Segment"] = segments

    return result


def get_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute segment-level summary statistics including customer counts, percentage share,
    and mean Recency, Frequency, and Monetary metrics.

    Parameters:
        df (pd.DataFrame):
            DataFrame containing `Segment` column, or score columns from which segments
            can be assigned, along with numeric metric columns (`Recency`, `Frequency`, `Monetary`).

    Returns:
        pd.DataFrame:
            Aggregated summary table sorted by Count descending with columns:
            ['Segment', 'Count', 'Percentage', 'MeanRecency', 'MeanFrequency', 'MeanMonetary'].

    Raises:
        ValueError:
            If df is None or missing required columns.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    summary_columns = [
        "Segment",
        "Count",
        "Percentage",
        "MeanRecency",
        "MeanFrequency",
        "MeanMonetary",
    ]

    if df.empty:
        return pd.DataFrame(columns=summary_columns)

    work_df = df.copy()
    if "Segment" not in work_df.columns:
        work_df = assign_segments(work_df)

    total_customers = len(work_df)

    # Detect metric columns or fallback to score columns
    rec_col = (
        "Recency" if "Recency" in work_df.columns
        else ("RecencyScore" if "RecencyScore" in work_df.columns else "R_Score")
    )
    freq_col = (
        "Frequency" if "Frequency" in work_df.columns
        else ("FrequencyScore" if "FrequencyScore" in work_df.columns else "F_Score")
    )
    mon_col = (
        "Monetary" if "Monetary" in work_df.columns
        else ("MonetaryScore" if "MonetaryScore" in work_df.columns else "M_Score")
    )

    grouped = work_df.groupby("Segment", observed=False)
    counts = grouped.size()
    percentages = (counts / total_customers * 100).round(2)

    mean_rec = grouped[rec_col].mean().round(2) if rec_col in work_df.columns else pd.Series(0.0, index=counts.index)
    mean_freq = grouped[freq_col].mean().round(2) if freq_col in work_df.columns else pd.Series(0.0, index=counts.index)
    mean_mon = grouped[mon_col].mean().round(2) if mon_col in work_df.columns else pd.Series(0.0, index=counts.index)

    summary = pd.DataFrame(
        {
            "Segment": counts.index,
            "Count": counts.values,
            "Percentage": percentages.values,
            "MeanRecency": mean_rec.values,
            "MeanFrequency": mean_freq.values,
            "MeanMonetary": mean_mon.values,
        }
    ).sort_values("Count", ascending=False).reset_index(drop=True)

    return summary


__all__ = [
    "assign_segments",
    "get_segment_summary",
    "extract_rfm_score_series",
    "SEGMENT_CHAMPIONS",
    "SEGMENT_CANT_LOSE",
    "SEGMENT_LOYAL",
    "SEGMENT_AT_RISK",
    "SEGMENT_NEW",
    "SEGMENT_PROMISING",
    "SEGMENT_NEED_ATTENTION",
    "SEGMENT_POTENTIAL_LOYAL",
    "SEGMENT_ABOUT_TO_SLEEP",
    "SEGMENT_LOST",
    "SEGMENT_HIBERNATING",
    "SEGMENT_OTHERS",
    "ALL_SEGMENTS",
]
