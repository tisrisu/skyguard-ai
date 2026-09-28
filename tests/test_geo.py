import json

from skyguard.config import MOCK_DIR
from skyguard.geo import haversine_km, neighbours


def test_haversine_delhi_to_mumbai():
    # roughly 1150 km in a straight line
    d = haversine_km(28.61, 77.21, 19.08, 72.88)
    assert 1100 < d < 1200


def test_neighbours_sorted_and_excludes_self():
    stations = json.loads((MOCK_DIR / "mock_stations.json").read_text(encoding="utf-8"))
    near = neighbours(stations, "DEL-01", radius_km=150)
    ids = [s["station_id"] for s in near]
    assert "DEL-01" not in ids
    assert [s["distance_km"] for s in near] == sorted(s["distance_km"] for s in near)
