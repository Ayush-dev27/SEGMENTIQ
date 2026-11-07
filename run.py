"""
SEGMENTIQ: Data-Driven Customer Segmentation & Retention Engine.
Main CLI Pipeline Orchestrator (Stage 7).

Executes the complete end-to-end customer analytics lifecycle:
1. Ingestion: Raw transactional Excel dataset loading & schema validation
2. Cleaning: Transaction filtering, deduplication, and sanitization
3. RFM Analytics: Behavioral Recency, Frequency, Monetary calculation & quantile scoring
4. Customer Segmentation: Rule-based heuristic lifecycle mapping & unsupervised K-Means clustering
5. Action Engine: Business strategy & marketing recommendation assignment
6. Export: Enriched customer-level dataset export to CSV
"""

import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd

from src.data.loader import load_data, DEFAULT_DATA_PATH
from src.data.cleaner import clean_data
from src.analytics.rfm import calculate_rfm_metrics
from src.analytics.scoring import assign_rfm_scores
from src.analytics.segmentation import assign_segments, get_segment_summary
from src.analytics.clustering import fit_kmeans_clusters, get_cluster_summary
from src.recommendations.actions import assign_recommendations, get_action_plan_summary

DEFAULT_OUTPUT_PATH = Path("data") / "processed" / "customers_rfm.csv"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format tabular data into a clean, aligned ASCII grid table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)

    formatted_rows = [
        " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{separator}\n" + "\n".join(formatted_rows)


def run_pipeline(
    raw_path: str = str(DEFAULT_DATA_PATH),
    output_path: str = str(DEFAULT_OUTPUT_PATH),
    n_clusters: int = 4,
    skip_clustering: bool = False,
) -> pd.DataFrame:
    """
    Execute the automated end-to-end SEGMENTIQ customer analytics pipeline.

    Parameters:
        raw_path (str): Filepath to the raw transactions dataset (.xlsx).
        output_path (str): Filepath to export the enriched customer dataset (.csv).
        n_clusters (int): Number of K-Means clusters to form. Default is 4.
        skip_clustering (bool): If True, bypasses K-Means clustering.

    Returns:
        pd.DataFrame: Fully processed and enriched customer-level DataFrame.
    """
    total_start_time = time.time()

    print("=" * 80)
    print("               SEGMENTIQ: CUSTOMER SEGMENTATION & RETENTION ENGINE               ")
    print("=" * 80)
    print(f"[*] Configuration:")
    print(f"    - Raw Data Source : {raw_path}")
    print(f"    - Output Target   : {output_path}")
    print(f"    - Target Clusters : {n_clusters} {'(Skipped)' if skip_clustering else ''}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: INGESTION
    # -------------------------------------------------------------------------
    step_start = time.time()
    print("\n[Stage 1/6] Ingestion: Loading raw transactional dataset...")
    raw_df = load_data(raw_path)
    step_time = time.time() - step_start
    print(f"    -> Ingested {len(raw_df):,} raw transactions across {raw_df.shape[1]} columns ({step_time:.2f}s)")

    # -------------------------------------------------------------------------
    # STAGE 2: CLEANING & SANITIZATION
    # -------------------------------------------------------------------------
    step_start = time.time()
    print("\n[Stage 2/6] Cleaning: Sanitizing transactions and enforcing data types...")
    cleaned_df = clean_data(raw_df)
    step_time = time.time() - step_start
    dropped_count = len(raw_df) - len(cleaned_df)
    print(f"    -> Filtered out {dropped_count:,} invalid rows ({dropped_count/len(raw_df)*100:.1f}%)")
    print(f"    -> Retained {len(cleaned_df):,} verified transactions ({step_time:.2f}s)")

    # -------------------------------------------------------------------------
    # STAGE 3: RFM CALCULATION & QUANTILE SCORING
    # -------------------------------------------------------------------------
    step_start = time.time()
    print("\n[Stage 3/6] Analytics: Computing customer-level RFM metrics & quantiles...")
    rfm_df = calculate_rfm_metrics(cleaned_df)
    scored_df = assign_rfm_scores(rfm_df)
    step_time = time.time() - step_start
    print(f"    -> Aggregated {len(scored_df):,} unique customers")
    print(f"    -> Generated RecencyScore, FrequencyScore, MonetaryScore, RFMScore, RFMTOTAL ({step_time:.2f}s)")

    # -------------------------------------------------------------------------
    # STAGE 4: RULE SEGMENTATION & ML CLUSTERING
    # -------------------------------------------------------------------------
    step_start = time.time()
    print("\n[Stage 4/6] Segmentation: Assigning heuristic lifecycle segments...")
    segmented_df = assign_segments(scored_df)

    if not skip_clustering:
        print(f"    -> Fitting unsupervised K-Means clustering (k={n_clusters})...")
        clustered_df, kmeans_model, scaler = fit_kmeans_clusters(
            segmented_df, n_clusters=n_clusters, random_state=42
        )
    else:
        print("    -> ML clustering bypassed by CLI flag.")
        clustered_df = segmented_df

    step_time = time.time() - step_start
    print(f"    -> Segmented {len(clustered_df):,} customers into behavioral cohorts ({step_time:.2f}s)")

    # -------------------------------------------------------------------------
    # STAGE 5: ACTION ENGINE & RECOMMENDATIONS
    # -------------------------------------------------------------------------
    step_start = time.time()
    print("\n[Stage 5/6] Action Engine: Attaching targeted e-commerce marketing strategies...")
    enriched_df = assign_recommendations(clustered_df, segment_col="Segment")
    step_time = time.time() - step_start
    print(f"    -> Attached tactical offers, priorities, and communication channels ({step_time:.2f}s)")

    # -------------------------------------------------------------------------
    # STAGE 6: DATA EXPORT
    # -------------------------------------------------------------------------
    step_start = time.time()
    print(f"\n[Stage 6/6] Export: Persisting customer intelligence to disk...")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    enriched_df.to_csv(output_path, index=False)
    file_size_bytes = os.path.getsize(output_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    step_time = time.time() - step_start
    print(f"    -> Successfully written to '{output_path}' ({file_size_mb:.2f} MB, {step_time:.2f}s)")

    total_time = time.time() - total_start_time

    # -------------------------------------------------------------------------
    # EXECUTIVE SUMMARY & REPORTING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                         EXECUTIVE PIPELINE SUMMARY                             ")
    print("=" * 80)
    print(f"[*] Total Execution Time      : {total_time:.2f} seconds")
    print(f"[*] Raw Transactions Ingested: {len(raw_df):,}")
    print(f"[*] Clean Transactions Retained: {len(cleaned_df):,} ({len(cleaned_df)/len(raw_df)*100:.1f}%)")
    print(f"[*] Unique Customers Mapped   : {len(enriched_df):,}")
    print(f"[*] Output Dataset File       : {output_path} ({file_size_mb:.2f} MB)")
    print("-" * 80)

    # 1. Segment Breakdown Table
    print("\n[+] CUSTOMER LIFECYCLE SEGMENT DISTRIBUTION:")
    segment_summary = get_segment_summary(enriched_df)
    seg_headers = ["Segment", "Count", "Share (%)", "Mean Recency (days)", "Mean Frequency", "Mean Monetary (£)"]
    seg_rows = [
        [
            row["Segment"],
            f"{int(row['Count']):,}",
            f"{row['Percentage']:.2f}%",
            f"{row['MeanRecency']:.1f}",
            f"{row['MeanFrequency']:.1f}",
            f"£{row['MeanMonetary']:,.2f}",
        ]
        for _, row in segment_summary.iterrows()
    ]
    print(format_table(seg_headers, seg_rows))

    # 2. Cluster Breakdown Table (if available)
    if "Cluster" in enriched_df.columns:
        print("\n[+] MACHINE LEARNING (K-MEANS) CLUSTER DISTRIBUTION:")
        cluster_summary = get_cluster_summary(enriched_df)
        clus_headers = ["Cluster ID", "Count", "Share (%)", "Mean Recency (days)", "Mean Frequency", "Mean Monetary (£)"]
        clus_rows = [
            [
                f"Cluster {int(row['Cluster'])}",
                f"{int(row['Count']):,}",
                f"{row['Percentage']:.2f}%",
                f"{row['Recency_Mean']:.1f}",
                f"{row['Frequency_Mean']:.1f}",
                f"£{row['Monetary_Mean']:,.2f}",
            ]
            for _, row in cluster_summary.iterrows()
        ]
        print(format_table(clus_headers, clus_rows))

    # 3. Action Plan Priorities
    print("\n[+] ACTION ENGINE: EXECUTIVE PRIORITY PLAN:")
    action_plan = get_action_plan_summary(enriched_df)
    act_headers = ["Priority", "Segment", "Count", "Action Title", "Primary Goal"]
    act_rows = [
        [
            row["Action_Priority"],
            row["Segment"],
            f"{int(row['Count']):,}",
            row["Action_Title"],
            (row["Action_Goal"][:45] + "...") if len(row["Action_Goal"]) > 48 else row["Action_Goal"],
        ]
        for _, row in action_plan.iterrows()
    ]
    print(format_table(act_headers, act_rows))

    print("\n" + "=" * 80)
    print("                 PIPELINE RUN COMPLETED SUCCESSFULLY! :white_check_mark:                 ")
    print("=" * 80 + "\n")

    return enriched_df


def main():
    parser = argparse.ArgumentParser(
        description="SEGMENTIQ: Automated Customer Segmentation & Action Engine CLI"
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the raw transactions dataset (.xlsx)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Target output CSV file path for enriched customer RFM dataset",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of K-Means clusters to generate (default: 4)",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip unsupervised K-Means clustering",
    )

    args = parser.parse_args()

    run_pipeline(
        raw_path=args.raw_path,
        output_path=args.output_path,
        n_clusters=args.clusters,
        skip_clustering=args.skip_clustering,
    )


if __name__ == "__main__":
    main()
