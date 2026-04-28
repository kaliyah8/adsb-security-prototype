# utils.py
from __future__ import annotations
import math
from typing import Tuple

EARTH_RADIUS_M = 6371008.8  # mean Earth radius in meters


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between (lat1, lon1) and (lat2, lon2) in METERS.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return float("nan")

    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def move_latlon(lat: float, lon: float, heading_deg: float, distance_m: float) -> Tuple[float, float]:
    """
    Move from (lat, lon) along heading_deg by distance_m over a sphere.
    Returns (new_lat, new_lon) in degrees.
    """
    lat1 = math.radians(float(lat))
    lon1 = math.radians(float(lon))
    brng = math.radians(float(heading_deg))
    dr = float(distance_m) / EARTH_RADIUS_M

    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(dr) * math.cos(lat1),
                             math.cos(dr) - math.sin(lat1) * math.sin(lat2))

    # normalize lon to [-180, 180)
    lon2_deg = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
    return math.degrees(lat2), lon2_deg