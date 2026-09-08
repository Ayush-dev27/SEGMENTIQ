"""
Unit tests for SEGMENTIQ data cleaner module (src/data/cleaner.py).
"""

import unittest
import numpy as np
import pandas as pd

from src.data.cleaner import (
    clean_data,
    remove_missing_customers,
    remove_cancellations,
    remove_invalid_quantities,
    remove_invalid_prices,
    remove_duplicates,
    handle_descriptions,
    enforce_datatypes,
    REQUIRED_COLUMNS,
)


class TestDataCleaner(unittest.TestCase):
    """Test suite for data cleaner functions and pipeline."""

    def _create_sample_row(self, **overrides) -> dict:
        """Helper to create a single valid transaction row with optional overrides."""
        row = {
            "InvoiceNo": 536365,
            "StockCode": "85123A",
            "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "Quantity": 6,
            "InvoiceDate": "2010-12-01 08:26:00",
            "UnitPrice": 2.55,
            "CustomerID": 17850.0,
            "Country": "United Kingdom",
        }
        row.update(overrides)
        return row

    def test_missing_customer_id_removal(self):
        """Verify that rows with null, NaN, or blank CustomerID are removed."""
        data = [
            self._create_sample_row(CustomerID=17850.0),
            self._create_sample_row(CustomerID=np.nan),
            self._create_sample_row(CustomerID=None),
            self._create_sample_row(CustomerID=""),
            self._create_sample_row(CustomerID="nan"),
            self._create_sample_row(CustomerID=13047.0),
        ]
        df = pd.DataFrame(data)
        cleaned = remove_missing_customers(df)

        self.assertEqual(len(cleaned), 2)
        self.assertListEqual(
            list(cleaned["CustomerID"].astype(float).astype(int)),
            [17850, 13047],
        )

    def test_cancellation_removal(self):
        """Verify that cancelled transactions (InvoiceNo starting with 'C' or 'c') are removed."""
        data = [
            self._create_sample_row(InvoiceNo=536365),
            self._create_sample_row(InvoiceNo="C536366"),
            self._create_sample_row(InvoiceNo="c536367"),
            self._create_sample_row(InvoiceNo="  C536368  "),
            self._create_sample_row(InvoiceNo="536369"),
        ]
        df = pd.DataFrame(data)
        cleaned = remove_cancellations(df)

        self.assertEqual(len(cleaned), 2)
        self.assertTrue(
            all(not str(inv).strip().upper().startswith("C") for inv in cleaned["InvoiceNo"])
        )

    def test_negative_and_zero_quantity_removal(self):
        """Verify that rows with Quantity <= 0 or non-numeric are removed."""
        data = [
            self._create_sample_row(Quantity=10),
            self._create_sample_row(Quantity=0),
            self._create_sample_row(Quantity=-5),
            self._create_sample_row(Quantity="invalid"),
            self._create_sample_row(Quantity=1),
        ]
        df = pd.DataFrame(data)
        cleaned = remove_invalid_quantities(df)

        self.assertEqual(len(cleaned), 2)
        self.assertTrue((cleaned["Quantity"].astype(int) > 0).all())

    def test_invalid_unit_price_removal(self):
        """Verify that rows with UnitPrice <= 0 or non-numeric are removed."""
        data = [
            self._create_sample_row(UnitPrice=4.50),
            self._create_sample_row(UnitPrice=0.0),
            self._create_sample_row(UnitPrice=-1.25),
            self._create_sample_row(UnitPrice="free"),
            self._create_sample_row(UnitPrice=0.01),
        ]
        df = pd.DataFrame(data)
        cleaned = remove_invalid_prices(df)

        self.assertEqual(len(cleaned), 2)
        self.assertTrue((cleaned["UnitPrice"].astype(float) > 0.0).all())

    def test_duplicate_removal(self):
        """Verify that exact duplicate rows across all columns are removed."""
        row1 = self._create_sample_row(InvoiceNo=536365, StockCode="A")
        row2 = self._create_sample_row(InvoiceNo=536365, StockCode="B")
        data = [row1, row1, row2, row1]
        df = pd.DataFrame(data)
        cleaned = remove_duplicates(df)

        self.assertEqual(len(cleaned), 2)

    def test_missing_description_handling(self):
        """Verify that missing/blank descriptions are normalized without inventing information."""
        data = [
            self._create_sample_row(Description="HAND WARMER"),
            self._create_sample_row(Description=None),
            self._create_sample_row(Description=np.nan),
            self._create_sample_row(Description="   "),
        ]
        df = pd.DataFrame(data)
        cleaned = handle_descriptions(df, fill_value="Unknown")

        self.assertEqual(cleaned.loc[0, "Description"], "HAND WARMER")
        self.assertEqual(cleaned.loc[1, "Description"], "Unknown")
        self.assertEqual(cleaned.loc[2, "Description"], "Unknown")
        self.assertEqual(cleaned.loc[3, "Description"], "Unknown")

    def test_correct_datatype_handling(self):
        """Verify that CustomerID is int64, InvoiceDate is datetime, Quantity is int64, UnitPrice is float64."""
        data = [
            self._create_sample_row(
                CustomerID="17850.0",
                InvoiceDate="2010-12-01 08:26:00",
                Quantity="12",
                UnitPrice="3.75",
            )
        ]
        df = pd.DataFrame(data)
        cleaned = enforce_datatypes(df)

        self.assertTrue(pd.api.types.is_integer_dtype(cleaned["CustomerID"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"]))
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned["Quantity"]))
        self.assertTrue(pd.api.types.is_float_dtype(cleaned["UnitPrice"]))
        self.assertEqual(cleaned["CustomerID"].iloc[0], 17850)

    def test_index_reset(self):
        """Verify that the DataFrame index is cleanly reset to a 0-indexed range."""
        data = [
            self._create_sample_row(InvoiceNo="C536365"),  # removed
            self._create_sample_row(InvoiceNo=536366),     # kept
            self._create_sample_row(Quantity=-1),          # removed
            self._create_sample_row(InvoiceNo=536368),     # kept
        ]
        df = pd.DataFrame(data)
        cleaned = clean_data(df)

        self.assertEqual(list(cleaned.index), [0, 1])

    def test_full_cleaning_pipeline(self):
        """End-to-end test verifying that all filter and transformation steps work together."""
        data = [
            self._create_sample_row(InvoiceNo=536365, CustomerID=17850.0),             # Valid 1
            self._create_sample_row(InvoiceNo="C536366", CustomerID=17850.0),           # Cancellation
            self._create_sample_row(InvoiceNo=536367, CustomerID=np.nan),              # Missing Customer
            self._create_sample_row(InvoiceNo=536368, Quantity=0),                      # Zero Quantity
            self._create_sample_row(InvoiceNo=536369, UnitPrice=0.0),                   # Zero UnitPrice
            self._create_sample_row(InvoiceNo=536370, CustomerID=13047.0),             # Valid 2
            self._create_sample_row(InvoiceNo=536370, CustomerID=13047.0),             # Duplicate of Valid 2
        ]
        df = pd.DataFrame(data)
        cleaned = clean_data(df)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(list(cleaned.index), [0, 1])
        self.assertEqual(cleaned["CustomerID"].tolist(), [17850, 13047])
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned["CustomerID"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"]))

    def test_missing_required_columns_raises_error(self):
        """Verify ValueError is raised if any required columns are missing."""
        df = pd.DataFrame({"InvoiceNo": [536365], "StockCode": ["85123A"]})
        with self.assertRaises(ValueError):
            clean_data(df)


if __name__ == "__main__":
    unittest.main()
