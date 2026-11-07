"""
API integration tests for SEGMENTIQ backend service (src/api/routes.py & main.py).
"""

import unittest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import get_processed_data


class TestAPI(unittest.TestCase):
    """Test suite for SEGMENTIQ FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient and verify sample data is available."""
        cls.client = TestClient(app)
        cls.df = get_processed_data()
        assert not cls.df.empty, "Processed customer dataset data/processed/customers_rfm.csv must exist for tests."
        cls.known_customer_id = int(cls.df.iloc[0]["CustomerID"])

    def test_health_check(self):
        """Test 1: Verify /api/v1/health returns status 200 and ok status."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("dataset_loaded"))
        self.assertGreater(data.get("total_customers", 0), 0)

    def test_get_summary(self):
        """Test 2: Validates aggregate figures, segment distributions, and cluster breakdowns."""
        response = self.client.get("/api/v1/summary")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("total_customers", data)
        self.assertIn("total_revenue", data)
        self.assertIn("avg_recency", data)
        self.assertIn("avg_frequency", data)
        self.assertIn("avg_monetary", data)
        self.assertIn("segment_distribution", data)
        self.assertIn("cluster_distribution", data)
        self.assertIn("priority_distribution", data)

        self.assertEqual(data["total_customers"], len(self.df))
        self.assertGreater(data["total_revenue"], 0)
        self.assertIsInstance(data["segment_distribution"], list)
        self.assertGreater(len(data["segment_distribution"]), 0)

    def test_get_customers_pagination(self):
        """Test 3: Verifies page, limit, total pages, and record counts."""
        limit = 10
        response = self.client.get(f"/api/v1/customers?page=1&limit={limit}")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["limit"], limit)
        self.assertEqual(data["total"], len(self.df))
        self.assertEqual(len(data["customers"]), limit)
        self.assertGreater(data["total_pages"], 1)

    def test_get_customers_filter_by_segment(self):
        """Test 4: Verifies filtering by segment (case-insensitive)."""
        response = self.client.get("/api/v1/customers?segment=champions&limit=50")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertGreater(data["total"], 0)
        for cust in data["customers"]:
            self.assertEqual(cust["Segment"].lower(), "champions")

    def test_get_single_customer_found(self):
        """Test 5: Fetches a known customer ID (status 200) with complete fields."""
        response = self.client.get(f"/api/v1/customers/{self.known_customer_id}")
        self.assertEqual(response.status_code, 200)

        cust = response.json()
        self.assertEqual(int(cust["CustomerID"]), self.known_customer_id)
        self.assertIn("Recency", cust)
        self.assertIn("Frequency", cust)
        self.assertIn("Monetary", cust)
        self.assertIn("Segment", cust)
        self.assertIn("Action_Title", cust)
        self.assertIn("Action_Priority", cust)

    def test_get_single_customer_not_found(self):
        """Test 6: Fetches non-existent customer ID (status 404)."""
        non_existent_id = 999999999
        response = self.client.get(f"/api/v1/customers/{non_existent_id}")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_root_endpoint(self):
        """Verify root endpoint returns metadata and documentation links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("SEGMENTIQ", response.json()["service"])


if __name__ == "__main__":
    unittest.main()
