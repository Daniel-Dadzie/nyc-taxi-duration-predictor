import importlib
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient


def _base_payload() -> dict:
    return {
        "pickup_longitude": -73.982155,
        "pickup_latitude": 40.767937,
        "dropoff_longitude": -73.964630,
        "dropoff_latitude": 40.765602,
        "passenger_count": 1,
        "vendor_id": 1,
        "pickup_datetime": "2016-06-12 00:43:35",
    }


class TestApiValidation(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        os.environ.setdefault("MODEL_REGISTRY_NAME", "group-a1-model")

        api_main = importlib.import_module("api.main")
        self.api_main = importlib.reload(api_main)

        # Skip external MLflow model loading for request-validation tests.
        self.load_model_patcher = mock.patch.object(
            self.api_main, "load_production_model", return_value=None
        )
        self.load_model_patcher.start()

        self.client = TestClient(self.api_main.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.load_model_patcher.stop()

    def test_predict_invalid_pickup_datetime_returns_422(self):
        payload = _base_payload()
        payload["pickup_datetime"] = "2016/06/12 00:43:35"

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("pickup_datetime", str(response.json()))

    def test_predict_invalid_vendor_id_returns_422(self):
        payload = _base_payload()
        payload["vendor_id"] = 3

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("vendor_id", str(response.json()))


if __name__ == "__main__":
    unittest.main()
