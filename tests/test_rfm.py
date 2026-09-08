"""
Unit tests for SEGMENTIQ RFM analytics module (src/analytics/rfm.py).
"""

import unittest
from datetime import datetime
import pandas as pd

from src.analytics.rfm import (
    calculate_rfm,
    determine_reference_date,
    validate_rfm_input,
)


class TestRFMAnalytics(unittest.TestCase):
    """Test suite for RFM calculation and validation."""

    def setUp(self):
        """Set up standard synthetic transaction data for testing."""
        self.sample_data = pd.DataFrame(
            {
                "InvoiceNo": [1001, 1001, 1002, 1003, 1004],
                "StockCode": ["A", "B", "C", "D", "E"],
                "Description": ["Item A", "Item B", "Item C", "Item D", "Item E"],
                "Quantity": [2, 3, 1, 5, 2],
                "InvoiceDate": pd.to_datetime(
                    [
                        "2023-01-01 10:00:00",
                        "2023-01-01 10:00:00",
                        "2023-01-05 12:00:00",
                        "2023-01-10 14:00:00",
                        "2023-01-15 16:00:00",
                    ]
                ),
                "UnitPrice": [10.0, 5.0, 20.0, 8.0, 50.0],
                "CustomerID": [101, 101, 101, 102, 103],
                "Country": ["United Kingdom"] * 5,
            }
        )

    def test_missing_required_columns_raises_error(self):
        """Verify that ValueError is raised if any required columns are missing."""
        invalid_df = pd.DataFrame(
            {"InvoiceNo": [1001], "Quantity": [2], "UnitPrice": [10.0]}
        )
        with self.assertRaises(ValueError):
            calculate_rfm(invalid_df)

    def test_default_reference_date(self):
        """Verify that default reference date is one day after the max InvoiceDate."""
        ref = determine_reference_date(self.sample_data)
        expected = pd.Timestamp("2023-01-16 16:00:00")
        self.assertEqual(ref, expected)

    def test_custom_reference_date(self):
        """Verify that a custom reference date is correctly applied."""
        custom_ref = "2023-01-20 00:00:00"
        ref = determine_reference_date(self.sample_data, reference_date=custom_ref)
        self.assertEqual(ref, pd.Timestamp("2023-01-20 00:00:00"))

    def test_rfm_metrics_calculation_default_date(self):
        """
        Verify RFM metrics for each customer using the default snapshot date.

        Max date in dataset is '2023-01-15 16:00:00'.
        Default snapshot is '2023-01-16 16:00:00'.

        Customer 101:
          - Latest InvoiceDate: 2023-01-05 12:00:00 -> Recency = 11 days
          - Unique Invoices: 1001, 1002 -> Frequency = 2
          - Monetary: (2*10.0 + 3*5.0) + (1*20.0) = 20 + 15 + 20 = 55.0

        Customer 102:
          - Latest InvoiceDate: 2023-01-10 14:00:00 -> Recency = 6 days
          - Unique Invoices: 1003 -> Frequency = 1
          - Monetary: 5 * 8.0 = 40.0

        Customer 103:
          - Latest InvoiceDate: 2023-01-15 16:00:00 -> Recency = 1 day
          - Unique Invoices: 1004 -> Frequency = 1
          - Monetary: 2 * 50.0 = 100.0
        """
        rfm = calculate_rfm(self.sample_data)

        self.assertEqual(len(rfm), 3)
        self.assertListEqual(
            list(rfm.columns), ["CustomerID", "Recency", "Frequency", "Monetary"]
        )

        c101 = rfm[rfm["CustomerID"] == 101].iloc[0]
        self.assertEqual(c101["Recency"], 11)
        self.assertEqual(c101["Frequency"], 2)
        self.assertEqual(c101["Monetary"], 55.0)

        c102 = rfm[rfm["CustomerID"] == 102].iloc[0]
        self.assertEqual(c102["Recency"], 6)
        self.assertEqual(c102["Frequency"], 1)
        self.assertEqual(c102["Monetary"], 40.0)

        c103 = rfm[rfm["CustomerID"] == 103].iloc[0]
        self.assertEqual(c103["Recency"], 1)
        self.assertEqual(c103["Frequency"], 1)
        self.assertEqual(c103["Monetary"], 100.0)

    def test_rfm_datatypes(self):
        """Verify that output column types are int64, int64, int64, float64."""
        rfm = calculate_rfm(self.sample_data)

        self.assertTrue(pd.api.types.is_integer_dtype(rfm["CustomerID"]))
        self.assertTrue(pd.api.types.is_integer_dtype(rfm["Recency"]))
        self.assertTrue(pd.api.types.is_integer_dtype(rfm["Frequency"]))
        self.assertTrue(pd.api.types.is_float_dtype(rfm["Monetary"]))

    def test_empty_dataframe(self):
        """Verify that passing an empty DataFrame returns an empty RFM DataFrame."""
        empty_df = pd.DataFrame(
            columns=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"]
        )
        rfm = calculate_rfm(empty_df)
        self.assertEqual(len(rfm), 0)
        self.assertListEqual(
            list(rfm.columns), ["CustomerID", "Recency", "Frequency", "Monetary"]
        )


if __name__ == "__main__":
    unittest.main()
