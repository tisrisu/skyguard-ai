"""Distance helpers for finding neighbouring stations."""

import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def neighbours(stations: list[dict], station_id: str, radius_km: float) -> list[dict]:
    """Stations within radius_km of station_id (excluding itself), nearest first.

    Each returned dict is a copy of the station entry with an added "distance_km".
    """
    home = next(s for s in stations if s["station_id"] == station_id)
    found = []
    for s in stations:
        if s["station_id"] == station_id:
            continue
        d = haversine_km(home["lat"], home["lon"], s["lat"], s["lon"])
        if d <= radius_km:
            found.append({**s, "distance_km": round(d, 1)})
    return sorted(found, key=lambda s: s["distance_km"])
