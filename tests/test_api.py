"""
tests/test_api.py — Group A1 — Taxi Trip Duration Predictor

API endpoint tests using FastAPI's TestClient (no running server needed).
Covers: /health, /model-info, /coverage, /predict, /predict-by-address, /reload-model

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── Fixtures: fake model + model_state so tests don't need MLflow ─────────────

MOCK_VERSION = "13"
MOCK_METRICS = {"primary_metric": 282.4, "rmse": 282.4, "rmsle": 0.305}

# Valid NYC coordinates (Times Square → JFK)
TIMES_SQUARE   = {"lat": 40.7580, "lon": -73.9855}
JFK            = {"lat": 40.6413, "lon": -73.7781}
NEWARK_AIRPORT = {"lat": 40.6895, "lon": -74.1745}

VALID_PREDICT_PAYLOAD = {
    "pickup_longitude":  TIMES_SQUARE["lon"],
    "pickup_latitude":   TIMES_SQUARE["lat"],
    "dropoff_longitude": JFK["lon"],
    "dropoff_latitude":  JFK["lat"],
    "pickup_datetime":   "2016-06-12 08:30:00",
}


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with the model pre-loaded via mocks.
    Patches MLflow and the model so no real registry is needed.
    """
    import numpy as np

    # Build a mock model that returns a log-space prediction (~847 seconds)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([np.log1p(847.0)])

    with (
        patch("api.main.load_production_model"),           # skip startup load
        patch("api.main.model_state", {
            "model":      mock_model,
            "version":    MOCK_VERSION,
            "stage":      "Production",
            "trained_at": 1686520800000,
            "metrics":    MOCK_METRICS,
        }),
    ):
        from api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ══════════════════════════════════════════════════════════════════════════════
# /health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_returns_ok_status(self, client):
        res = client.get("/health")
        assert res.json() == {"status": "ok"}

    def test_is_fast(self, client):
        import time
        start = time.time()
        client.get("/health")
        assert time.time() - start < 1.0   # well under 100 ms in CI


# ══════════════════════════════════════════════════════════════════════════════
# /model-info
# ══════════════════════════════════════════════════════════════════════════════

class TestModelInfo:
    def test_returns_200(self, client):
        res = client.get("/model-info")
        assert res.status_code == 200

    def test_response_shape(self, client):
        data = client.get("/model-info").json()
        assert "model_name"           in data
        assert "version"              in data
        assert "stage"                in data
        assert "trained_at"           in data
        assert "primary_metric"       in data
        assert "primary_metric_value" in data
        assert "model_loaded"         in data

    def test_model_loaded_is_true(self, client):
        data = client.get("/model-info").json()
        assert data["model_loaded"] is True

    def test_version_matches_mock(self, client):
        data = client.get("/model-info").json()
        assert data["version"] == MOCK_VERSION

    def test_stage_is_production(self, client):
        data = client.get("/model-info").json()
        assert data["stage"] == "Production"

    def test_primary_metric_label(self, client):
        data = client.get("/model-info").json()
        assert data["primary_metric"] == "RMSE"

    def test_primary_metric_value(self, client):
        data = client.get("/model-info").json()
        assert data["primary_metric_value"] == pytest.approx(282.4, rel=1e-3)

    def test_503_when_model_not_loaded(self, client):
        with patch("api.main.model_state", {**{
            "model": None, "version": None, "stage": None,
            "trained_at": None, "metrics": {},
        }}):
            res = client.get("/model-info")
            assert res.status_code == 503


# ══════════════════════════════════════════════════════════════════════════════
# /coverage
# ══════════════════════════════════════════════════════════════════════════════

class TestCoverage:
    def test_returns_200(self, client):
        res = client.get("/coverage")
        assert res.status_code == 200

    def test_has_boroughs(self, client):
        data = client.get("/coverage").json()
        assert "boroughs" in data
        assert "Manhattan" in data["boroughs"]

    def test_has_three_airports(self, client):
        data = client.get("/coverage").json()
        assert len(data["airports_supported"]) == 3

    def test_has_bounding_box(self, client):
        data = client.get("/coverage").json()
        bb = data["bounding_box"]
        assert bb["longitude"]["min"] == -74.25
        assert bb["longitude"]["max"] == -73.70
        assert bb["latitude"]["min"]  ==  40.49
        assert bb["latitude"]["max"]  ==  40.92

    def test_newark_is_nearby_supported(self, client):
        data = client.get("/coverage").json()
        nearby = " ".join(data["nearby_areas_supported"])
        assert "Newark" in nearby


# ══════════════════════════════════════════════════════════════════════════════
# /predict  — happy path
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictHappyPath:
    def test_returns_200(self, client):
        res = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
        assert res.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        assert "trip_duration_seconds" in data
        assert "trip_duration_minutes" in data
        assert "model_version"         in data

    def test_duration_is_positive(self, client):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        assert data["trip_duration_seconds"] > 0
        assert data["trip_duration_minutes"] > 0

    def test_minutes_equals_seconds_divided_by_60(self, client):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        assert data["trip_duration_minutes"] == pytest.approx(
            data["trip_duration_seconds"] / 60, rel=1e-2
        )

    def test_model_version_in_response(self, client):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        assert data["model_version"] == MOCK_VERSION

    def test_duration_within_realistic_range(self, client):
        data = client.post("/predict", json=VALID_PREDICT_PAYLOAD).json()
        # Clamped range: 60 s – 10 800 s
        assert 60 <= data["trip_duration_seconds"] <= 10_800

    def test_optional_datetime_omitted(self, client):
        """Omitting pickup_datetime should default to now and still return 200."""
        payload = {k: v for k, v in VALID_PREDICT_PAYLOAD.items()
                   if k != "pickup_datetime"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_iso_datetime_with_T_separator(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "pickup_datetime": "2016-06-12T08:30:00"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_newark_pickup(self, client):
        """Newark Airport pickup should be accepted as a special case."""
        payload = {
            **VALID_PREDICT_PAYLOAD,
            "pickup_longitude": NEWARK_AIRPORT["lon"],
            "pickup_latitude":  NEWARK_AIRPORT["lat"],
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    def test_newark_dropoff(self, client):
        """Newark Airport dropoff should be accepted as a special case."""
        payload = {
            **VALID_PREDICT_PAYLOAD,
            "dropoff_longitude": NEWARK_AIRPORT["lon"],
            "dropoff_latitude":  NEWARK_AIRPORT["lat"],
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# /predict  — validation errors (422)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictValidation:
    def test_pickup_longitude_out_of_bounds(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "pickup_longitude": -70.0}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_pickup_latitude_out_of_bounds(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "pickup_latitude": 0.0}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_dropoff_longitude_out_of_bounds(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "dropoff_longitude": -70.0}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_dropoff_latitude_out_of_bounds(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "dropoff_latitude": 0.0}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_bad_datetime_format(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "pickup_datetime": "12-06-2016 08:30"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_missing_required_field(self, client):
        payload = {k: v for k, v in VALID_PREDICT_PAYLOAD.items()
                   if k != "pickup_longitude"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_wrong_type_for_latitude(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "pickup_latitude": "not-a-number"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_validation_error_response_shape(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "pickup_longitude": -70.0}
        data = client.post("/predict", json=payload).json()
        # Custom handler returns error + details + hint
        assert "error"   in data or "detail" in data

    def test_empty_body(self, client):
        res = client.post("/predict", json={})
        assert res.status_code == 422

    def test_los_angeles_coordinates_rejected(self, client):
        """Coordinates from outside NYC should be rejected."""
        payload = {
            "pickup_longitude":  -118.2437,   # LA
            "pickup_latitude":    34.0522,
            "dropoff_longitude": -118.2437,
            "dropoff_latitude":   34.0522,
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# /predict-by-address
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictByAddress:
    """
    These tests mock geocode_address so they don't hit the real Nominatim API.
    """

    def _mock_location(self, lat, lon):
        loc = MagicMock()
        loc.latitude  = lat
        loc.longitude = lon
        return loc

    def test_valid_addresses_return_200(self, client):
        pickup_loc  = self._mock_location(TIMES_SQUARE["lat"], TIMES_SQUARE["lon"])
        dropoff_loc = self._mock_location(JFK["lat"], JFK["lon"])
        with patch("api.main.geocode_address", side_effect=[pickup_loc, dropoff_loc]):
            res = client.post("/predict-by-address", json={
                "pickup_address":  "Times Square, New York",
                "dropoff_address": "JFK Airport, Queens, NY",
            })
        assert res.status_code == 200

    def test_response_has_duration_fields(self, client):
        pickup_loc  = self._mock_location(TIMES_SQUARE["lat"], TIMES_SQUARE["lon"])
        dropoff_loc = self._mock_location(JFK["lat"], JFK["lon"])
        with patch("api.main.geocode_address", side_effect=[pickup_loc, dropoff_loc]):
            data = client.post("/predict-by-address", json={
                "pickup_address":  "Times Square, New York",
                "dropoff_address": "JFK Airport, Queens, NY",
            }).json()
        assert "trip_duration_seconds" in data
        assert "trip_duration_minutes" in data

    def test_address_not_found_returns_422(self, client):
        with patch("api.main.geocode_address", return_value=None):
            res = client.post("/predict-by-address", json={
                "pickup_address":  "Flushng, New York",   # intentional typo
                "dropoff_address": "JFK Airport, Queens, NY",
            })
        assert res.status_code == 422

    def test_outside_nyc_address_returns_422(self, client):
        with patch("api.main.geocode_address", return_value=None):
            res = client.post("/predict-by-address", json={
                "pickup_address":  "Los Angeles, CA",
                "dropoff_address": "JFK Airport, Queens, NY",
            })
        assert res.status_code == 422

    def test_empty_pickup_address_returns_422(self, client):
        res = client.post("/predict-by-address", json={
            "pickup_address":  "",
            "dropoff_address": "JFK Airport, Queens, NY",
        })
        assert res.status_code == 422

    def test_too_short_address_returns_422(self, client):
        res = client.post("/predict-by-address", json={
            "pickup_address":  "NY",   # too short (< 3 chars after strip... "NY" is 2)
            "dropoff_address": "JFK Airport, Queens, NY",
        })
        assert res.status_code == 422

    def test_missing_dropoff_address_returns_422(self, client):
        res = client.post("/predict-by-address", json={
            "pickup_address": "Times Square, New York",
        })
        assert res.status_code == 422

    def test_datetime_with_T_separator_accepted(self, client):
        pickup_loc  = self._mock_location(TIMES_SQUARE["lat"], TIMES_SQUARE["lon"])
        dropoff_loc = self._mock_location(JFK["lat"], JFK["lon"])
        with patch("api.main.geocode_address", side_effect=[pickup_loc, dropoff_loc]):
            res = client.post("/predict-by-address", json={
                "pickup_address":  "Times Square, New York",
                "dropoff_address": "JFK Airport, Queens, NY",
                "pickup_datetime": "2016-06-12T08:30:00",
            })
        assert res.status_code == 200

    def test_newark_airport_address_accepted(self, client):
        pickup_loc  = self._mock_location(TIMES_SQUARE["lat"], TIMES_SQUARE["lon"])
        dropoff_loc = self._mock_location(NEWARK_AIRPORT["lat"], NEWARK_AIRPORT["lon"])
        with patch("api.main.geocode_address", side_effect=[pickup_loc, dropoff_loc]):
            res = client.post("/predict-by-address", json={
                "pickup_address":  "Times Square, New York",
                "dropoff_address": "Newark Liberty International Airport, Newark, NJ",
            })
        assert res.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# /reload-model
# ══════════════════════════════════════════════════════════════════════════════

class TestReloadModel:
    def test_returns_200(self, client):
        with patch("api.main.load_production_model"):
            res = client.post("/reload-model")
        assert res.status_code == 200

    def test_returns_reloading_status(self, client):
        with patch("api.main.load_production_model"):
            data = client.post("/reload-model").json()
        assert data == {"status": "reloading"}