"""
Unit tests for business recommendations and action engine (src/recommendations/actions.py).
"""

import unittest
import pandas as pd

from src.recommendations.actions import (
    recommend_actions_for_customer,
    assign_recommendations,
    get_action_plan_summary,
    SEGMENT_STRATEGIES,
)


class TestActions(unittest.TestCase):
    """Test suite for customer action engine and business strategy mapping."""

    def setUp(self):
        """Set up standard sample DataFrames with customer segments."""
        self.sample_df = pd.DataFrame(
            {
                "CustomerID": [101, 102, 103, 104, 105, 106],
                "Segment": [
                    "Champions",
                    "At Risk",
                    "Loyal Customers",
                    "New Customers",
                    "Hibernating",
                    "Can't Lose Them",
                ],
            }
        )

    def test_all_segments_covered(self):
        """Test 1: Verify every predefined segment in SEGMENT_STRATEGIES has valid non-empty fields."""
        required_fields = [
            "action_title",
            "priority",
            "channel",
            "recommended_offers",
            "business_goal",
        ]
        valid_priorities = {"Urgent", "High", "Medium", "Low"}

        for segment_name, strategy in SEGMENT_STRATEGIES.items():
            for field in required_fields:
                self.assertIn(
                    field,
                    strategy,
                    f"Segment '{segment_name}' missing field '{field}'",
                )
                self.assertTrue(
                    strategy[field],
                    f"Segment '{segment_name}' has empty value for '{field}'",
                )

            # Check valid priority level
            self.assertIn(strategy["priority"], valid_priorities)

            # Check channels and offers are non-empty lists
            self.assertIsInstance(strategy["channel"], list)
            self.assertGreater(len(strategy["channel"]), 0)
            self.assertIsInstance(strategy["recommended_offers"], list)
            self.assertGreater(len(strategy["recommended_offers"]), 0)

    def test_single_customer_recommendation(self):
        """Test 2: Valid lookup for known segments and case/whitespace insensitivity."""
        # Exact match
        strat_champ = recommend_actions_for_customer("Champions")
        self.assertEqual(strat_champ["priority"], "High")
        self.assertIn("VIP", strat_champ["action_title"])

        # Case-insensitive & trimmed match
        strat_at_risk = recommend_actions_for_customer("  at risk  ")
        self.assertEqual(strat_at_risk["priority"], "Urgent")
        self.assertIn("Win-Back", strat_at_risk["action_title"])

        # Alternate naming check
        strat_new = recommend_actions_for_customer("recent customers")
        self.assertEqual(
            strat_new["action_title"],
            recommend_actions_for_customer("New Customers")["action_title"],
        )

    def test_unknown_segment_fallback(self):
        """Test 3: Check fallback to 'Others' on unrecognized segment strings or invalid types."""
        fallback_expected = SEGMENT_STRATEGIES["Others"]

        # Unknown string
        strat_unknown = recommend_actions_for_customer("Alien Segment")
        self.assertEqual(strat_unknown["action_title"], fallback_expected["action_title"])
        self.assertEqual(strat_unknown["priority"], "Low")

        # None input
        strat_none = recommend_actions_for_customer(None)
        self.assertEqual(strat_none["action_title"], fallback_expected["action_title"])

        # Empty string
        strat_empty = recommend_actions_for_customer("   ")
        self.assertEqual(strat_empty["action_title"], fallback_expected["action_title"])

    def test_assign_recommendations_dataframe(self):
        """Test 4: Check proper column creation, correct row count, and immutability of input DataFrame."""
        original_copy = self.sample_df.copy()
        original_cols = list(self.sample_df.columns)

        result_df = assign_recommendations(self.sample_df, segment_col="Segment")

        # Check immutability
        self.assertNotIn("Action_Title", self.sample_df.columns)
        self.assertListEqual(list(self.sample_df.columns), original_cols)
        pd.testing.assert_frame_equal(self.sample_df, original_copy)
        self.assertIsNot(result_df, self.sample_df)

        # Check output structure
        expected_cols = [
            "Action_Title",
            "Action_Priority",
            "Action_Goal",
            "Recommended_Channels",
        ]
        for col in expected_cols:
            self.assertIn(col, result_df.columns)

        self.assertEqual(len(result_df), len(self.sample_df))

        # Check values match mapped priorities
        self.assertEqual(result_df.loc[0, "Action_Priority"], "High")  # Champions
        self.assertEqual(result_df.loc[1, "Action_Priority"], "Urgent")  # At Risk

    def test_missing_segment_column_raises_error(self):
        """Test 5: Verify ValueError when segment_col is not found."""
        invalid_df = pd.DataFrame({"CustomerID": [1, 2], "WrongCol": ["A", "B"]})

        with self.assertRaises(ValueError):
            assign_recommendations(invalid_df, segment_col="Segment")

        with self.assertRaises(ValueError):
            assign_recommendations(None)

    def test_get_action_plan_summary(self):
        """Test 6: Ensure schema contains count, percentage, priority, actions, and handles empty DataFrame."""
        summary = get_action_plan_summary(self.sample_df, segment_col="Segment")

        expected_columns = [
            "Segment",
            "Count",
            "Percentage",
            "Action_Priority",
            "Action_Title",
            "Action_Goal",
        ]
        self.assertListEqual(list(summary.columns), expected_columns)

        # Total counts match input rows
        self.assertEqual(summary["Count"].sum(), len(self.sample_df))

        # Percentage sum ~ 100%
        self.assertAlmostEqual(summary["Percentage"].sum(), 100.0, places=1)

        # Check priority sorting: 'Urgent' must appear before 'High', which appears before 'Medium' / 'Low'
        priorities = summary["Action_Priority"].tolist()
        rank_order = {"Urgent": 0, "High": 1, "Medium": 2, "Low": 3}
        numerical_ranks = [rank_order.get(p, 99) for p in priorities]
        for i in range(len(numerical_ranks) - 1):
            self.assertLessEqual(
                numerical_ranks[i],
                numerical_ranks[i + 1],
                f"Priority order violated: {priorities[i]} before {priorities[i+1]}",
            )

        # Empty DataFrame handling
        empty_df = pd.DataFrame(columns=["Segment"])
        empty_summary = get_action_plan_summary(empty_df, segment_col="Segment")
        self.assertTrue(empty_summary.empty)
        self.assertListEqual(list(empty_summary.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()
