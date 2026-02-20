# data_simulation.py
import numpy as np
import pandas as pd
from utils import move_latlon

def _rand_icao(i: int) -> str:
    return f"SIM{i:04d}"

def generate_aircraft_data(
    num_aircraft: int = 25,
    time_steps: int = 120,
    center_lat: float = 33.9416,
    center_lon: float = -118.4085,
    dt_s: float = 1.0,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Physically consistent simulator in SI units:
      velocity: m/s, altitude: m, vertical_rate: m/s, heading: deg.
      timestamp: integer seconds.
    """
    rng = np.random.default_rng(seed)
    t0 = 1700000000

    rows = []
    for i in range(num_aircraft):
        icao = _rand_icao(i)

        lat = center_lat + rng.normal(0, 0.06)
        lon = center_lon + rng.normal(0, 0.06)

        v = float(rng.uniform(90.0, 250.0))       # m/s
        heading = float(rng.uniform(0.0, 360.0))  # deg
        alt = float(rng.uniform(500.0, 11000.0))  # m

        accel = float(rng.normal(0.0, 0.2))       # m/s^2
        turn_rate = float(rng.normal(0.0, 0.6))   # deg/s
        climb_rate = float(rng.normal(0.0, 0.5))  # m/s

        for k in range(time_steps):
            ts = t0 + k

            rows.append({
                "icao": icao,
                "timestamp": int(ts),
                "latitude": float(lat),
                "longitude": float(lon),
                "velocity": float(v),
                "heading": float(heading),
                "baro_altitude": float(alt),
                "vertical_rate": float(climb_rate),
            })

            # integrate
            v = max(0.0, v + accel * dt_s + rng.normal(0.0, 0.15))
            heading = (heading + turn_rate * dt_s + rng.normal(0.0, 0.05)) % 360.0
            alt = max(0.0, alt + climb_rate * dt_s + rng.normal(0.0, 0.2))

            distance = v * dt_s
            lat, lon = move_latlon(lat, lon, heading, distance)

    return pd.DataFrame(rows)


def inject_teleportation_attack(df: pd.DataFrame, target_icao: str, start_step: int, end_step: int) -> pd.DataFrame:
    d = df.copy()
    m = (d["icao"].astype(str) == str(target_icao))
    if not m.any():
        return d

    idx = d[m].sort_values("timestamp").index.to_list()
    start_step = max(0, min(start_step, len(idx) - 1))
    end_step = max(start_step, min(end_step, len(idx) - 1))

    # big jump (~200km)
    jump_lat = 1.8
    jump_lon = 1.8

    attack_idx = idx[start_step:end_step + 1]
    d.loc[attack_idx, "latitude"] = d.loc[attack_idx, "latitude"] + jump_lat
    d.loc[attack_idx, "longitude"] = d.loc[attack_idx, "longitude"] + jump_lon
    return d


def inject_gps_spoofing_attack(
    df: pd.DataFrame,
    target_icao: str,
    start_step: int,
    end_step: int,
    lat_shift: float = 0.10,
    lon_shift: float = 0.10,
) -> pd.DataFrame:
    """
    Spoofing = persistent bias in position with unchanged velocity/heading.
    """
    d = df.copy()
    m = (d["icao"].astype(str) == str(target_icao))
    if not m.any():
        return d

    idx = d[m].sort_values("timestamp").index.to_list()
    start_step = max(0, min(start_step, len(idx) - 1))
    end_step = max(start_step, min(end_step, len(idx) - 1))

    attack_idx = idx[start_step:end_step + 1]
    d.loc[attack_idx, "latitude"] = d.loc[attack_idx, "latitude"] + float(lat_shift)
    d.loc[attack_idx, "longitude"] = d.loc[attack_idx, "longitude"] + float(lon_shift)
    return d


def inject_ghost_aircraft_attack(df: pd.DataFrame, ghost_icao: str, start_step: int, end_step: int) -> pd.DataFrame:
    """
    Ghost = add a new aircraft that appears suddenly then disappears.
    """
    d = df.copy()
    if d.empty:
        return d

    base = d.sort_values(["icao", "timestamp"]).copy()
    ts_unique = sorted(base["timestamp"].unique().tolist())
    if not ts_unique:
        return d

    start_step = max(0, min(start_step, len(ts_unique) - 1))
    end_step = max(start_step, min(end_step, len(ts_unique) - 1))

    attack_ts = ts_unique[start_step:end_step + 1]

    lat0 = float(base["latitude"].mean())
    lon0 = float(base["longitude"].mean())

    rows = []
    for ts in attack_ts:
        rows.append({
            "icao": str(ghost_icao),
            "timestamp": int(ts),
            "latitude": lat0 + 0.2,
            "longitude": lon0 - 0.2,
            "velocity": 0.0,
            "heading": 0.0,
            "baro_altitude": 5000.0,
            "vertical_rate": 0.0,
        })

    ghost_df = pd.DataFrame(rows)
    return pd.concat([d, ghost_df], ignore_index=True)