"""
Unit tests for customer segmentation module (src/analytics/segmentation.py).
"""

import unittest
import pandas as pd

from src.analytics.segmentation import (
    assign_segments,
    get_segment_summary,
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


class TestSegmentation(unittest.TestCase):
    """Test suite for RFM rule-based customer segmentation and summary reporting."""

    def setUp(self):
        """Set up standard test DataFrames with varied RFM score conventions."""
        self.sample_standard_df = pd.DataFrame(
            {
                "CustomerID": [101, 102, 103, 104, 105],
                "Recency": [5, 25, 75, 120, 350],
                "Frequency": [20, 8, 3, 1, 1],
                "Monetary": [4500.0, 1200.0, 350.0, 80.0, 25.0],
                "RecencyScore": [5, 4, 3, 4, 1],
                "FrequencyScore": [5, 4, 2, 1, 1],
                "MonetaryScore": [5, 4, 2, 1, 1],
                "RFMScore": ["555", "444", "322", "411", "111"],
            }
        )

    def test_extreme_cases_assignment(self):
        """Test 1: Valid assignment of extreme cases (Champions for 555, Lost for 111)."""
        df = pd.DataFrame(
            {
                "CustomerID": [1, 2],
                "RecencyScore": [5, 1],
                "FrequencyScore": [5, 1],
                "MonetaryScore": [5, 1],
            }
        )
        segmented = assign_segments(df)

        self.assertEqual(segmented.loc[0, "Segment"], SEGMENT_CHAMPIONS)
        self.assertEqual(segmented.loc[1, "Segment"], SEGMENT_LOST)

        # Also verify using string RFMScore
        df_str = pd.DataFrame({"CustomerID": [1, 2], "RFMScore": ["555", "111"]})
        segmented_str = assign_segments(df_str)
        self.assertEqual(segmented_str.loc[0, "Segment"], SEGMENT_CHAMPIONS)
        self.assertEqual(segmented_str.loc[1, "Segment"], SEGMENT_LOST)

    def test_intermediate_boundary_cases(self):
        """Test 2: Intermediate boundary cases (At Risk, Potential Loyalists, New Customers)."""
        df = pd.DataFrame(
            {
                "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8],
                "RecencyScore": [1, 4, 5, 1, 3, 3, 2, 2],
                "FrequencyScore": [3, 2, 1, 5, 1, 4, 2, 1],
                "MonetaryScore": [3, 2, 1, 5, 1, 4, 2, 1],
            }
        )
        segmented = assign_segments(df)

        # Customer 1: R=1, F=3, M=3 -> At Risk
        self.assertEqual(segmented.loc[0, "Segment"], SEGMENT_AT_RISK)

        # Customer 2: R=4, F=2, M=2 -> Potential Loyalists
        self.assertEqual(segmented.loc[1, "Segment"], SEGMENT_POTENTIAL_LOYAL)

        # Customer 3: R=5, F=1, M=1 -> New Customers
        self.assertEqual(segmented.loc[2, "Segment"], SEGMENT_NEW)

        # Customer 4: R=1, F=5, M=5 -> Can't Lose Them
        self.assertEqual(segmented.loc[3, "Segment"], SEGMENT_CANT_LOSE)

        # Customer 5: R=3, F=1, M=1 -> Promising
        self.assertEqual(segmented.loc[4, "Segment"], SEGMENT_PROMISING)

        # Customer 6: R=2, F=4, M=4 -> Loyal Customers
        self.assertEqual(segmented.loc[5, "Segment"], SEGMENT_LOYAL)

        # Customer 7: R=2, F=2, M=2 -> Customers Needing Attention
        self.assertEqual(segmented.loc[6, "Segment"], SEGMENT_NEED_ATTENTION)

        # Customer 8: R=2, F=1, M=1 -> About to Sleep
        self.assertEqual(segmented.loc[7, "Segment"], SEGMENT_ABOUT_TO_SLEEP)

    def test_alternate_naming_conventions(self):
        """Verify support for R_Score / F_Score / M_Score and R / F / M columns."""
        df_rscore = pd.DataFrame(
            {"CustomerID": [1, 2], "R_Score": [5, 1], "F_Score": [5, 1], "M_Score": [5, 1]}
        )
        segmented = assign_segments(df_rscore)
        self.assertEqual(segmented.loc[0, "Segment"], SEGMENT_CHAMPIONS)
        self.assertEqual(segmented.loc[1, "Segment"], SEGMENT_LOST)

        df_rfm = pd.DataFrame(
            {"CustomerID": [1, 2], "R": [5, 1], "F": [5, 1], "M": [5, 1]}
        )
        segmented2 = assign_segments(df_rfm)
        self.assertEqual(segmented2.loc[0, "Segment"], SEGMENT_CHAMPIONS)
        self.assertEqual(segmented2.loc[1, "Segment"], SEGMENT_LOST)

    def test_dataframe_immutability(self):
        """Test 3: DataFrame immutability (original input DataFrame remains unchanged)."""
        input_copy = self.sample_standard_df.copy()
        original_cols = list(self.sample_standard_df.columns)

        result = assign_segments(self.sample_standard_df)

        # Original input must not be modified
        self.assertNotIn("Segment", self.sample_standard_df.columns)
        self.assertListEqual(list(self.sample_standard_df.columns), original_cols)
        pd.testing.assert_frame_equal(self.sample_standard_df, input_copy)

        # Returned DataFrame must be a new object containing Segment
        self.assertIsNot(result, self.sample_standard_df)
        self.assertIn("Segment", result.columns)

    def test_missing_columns_validation(self):
        """Test 4: Missing columns validation (raises ValueError)."""
        invalid_df = pd.DataFrame({"CustomerID": [1, 2], "Unrelated": [10, 20]})

        with self.assertRaises(ValueError):
            assign_segments(invalid_df)

        with self.assertRaises(ValueError):
            assign_segments(None)

    def test_get_segment_summary(self):
        """Test 5: get_segment_summary returns correct aggregated metrics, column names, and row counts."""
        segmented = assign_segments(self.sample_standard_df)
        summary = get_segment_summary(segmented)

        expected_columns = [
            "Segment",
            "Count",
            "Percentage",
            "MeanRecency",
            "MeanFrequency",
            "MeanMonetary",
        ]
        self.assertListEqual(list(summary.columns), expected_columns)

        # Sum of counts equals total customers in input
        self.assertEqual(summary["Count"].sum(), len(self.sample_standard_df))

        # Sum of percentages should be ~100%
        self.assertAlmostEqual(summary["Percentage"].sum(), 100.0, places=1)

        # Row counts match number of unique segments present
        self.assertEqual(len(summary), segmented["Segment"].nunique())

        # Verify summary works even if called directly on unsegmented scored df
        summary_direct = get_segment_summary(self.sample_standard_df)
        self.assertListEqual(list(summary_direct.columns), expected_columns)
        self.assertEqual(summary_direct["Count"].sum(), len(self.sample_standard_df))

    def test_empty_dataframe_handling(self):
        """Test 6: Empty DataFrame handling."""
        empty_df = pd.DataFrame(columns=["CustomerID", "RecencyScore", "FrequencyScore", "MonetaryScore"])
        segmented = assign_segments(empty_df)

        self.assertTrue(segmented.empty)
        self.assertIn("Segment", segmented.columns)

        summary = get_segment_summary(segmented)
        self.assertTrue(summary.empty)
        self.assertIn("Segment", summary.columns)
        self.assertIn("Count", summary.columns)
        self.assertIn("Percentage", summary.columns)

    def test_fallback_assignment(self):
        """Verify unassigned edge cases cleanly fall back to fallback bucket."""
        # Value where F and M diverge widely and aren't caught by standard rules
        unusual_df = pd.DataFrame(
            {"CustomerID": [999], "RecencyScore": [1], "FrequencyScore": [1], "MonetaryScore": [5]}
        )
        segmented = assign_segments(unusual_df, fallback=SEGMENT_OTHERS)
        self.assertEqual(segmented.loc[0, "Segment"], SEGMENT_OTHERS)


if __name__ == "__main__":
    unittest.main()
