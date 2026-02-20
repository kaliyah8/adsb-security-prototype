# data_fetcher.py
import os
import time
from typing import Tuple

import pandas as pd
import requests

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


def fetch_live_adsb_data(
    lamin: float,
    lomin: float,
    lamax: float,
    lomax: float,
    timeout: int = 12,
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch OpenSky state vectors within bbox.
    Returns (df, err_string). df uses OpenSky column names.

    Units from OpenSky:
      velocity: m/s
      baro_altitude: m
      vertical_rate: m/s
      true_track: degrees
      time_position / last_contact: epoch seconds
    """
    # Normalize bbox
    if lamax < lamin:
        lamin, lamax = lamax, lamin
    if lomax < lomin:
        lomin, lomax = lomax, lomin

    params = {"lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax}
    headers = {}

    cid = (os.getenv("OPENSKY_CLIENT_ID") or "").strip()
    secret = (os.getenv("OPENSKY_CLIENT_SECRET") or "").strip()

    # Optional OAuth
    if cid and secret:
        try:
            token_resp = requests.post(
                OPENSKY_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": cid,
                    "client_secret": secret,
                },
                timeout=timeout,
            )
            if token_resp.status_code == 200:
                token = token_resp.json().get("access_token")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
        except Exception:
            pass  # fallback to unauthenticated

    try:
        r = requests.get(OPENSKY_STATES_URL, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return pd.DataFrame(), f"OpenSky HTTP {r.status_code}: {r.text[:200]}"

        js = r.json()
        states = js.get("states") or []
        if not states:
            return pd.DataFrame(), "OpenSky returned 0 states."

        columns = [
            "icao24", "callsign", "origin_country", "time_position",
            "last_contact", "lon", "lat", "baro_altitude", "on_ground",
            "velocity", "true_track", "vertical_rate", "sensors",
            "geo_altitude", "squawk", "spi", "position_source"
        ]

        rows = []
        for s in states:
            s2 = list(s) + [None] * (len(columns) - len(s))
            rows.append(s2[:len(columns)])

        df = pd.DataFrame(rows, columns=columns)

        df["callsign"] = df["callsign"].astype(str).str.strip().replace({"None": ""})

        num_cols = ["lon", "lat", "baro_altitude", "velocity", "true_track", "vertical_rate", "geo_altitude", "last_contact", "time_position"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["fetched_at_epoch"] = int(time.time())
        return df, ""

    except requests.exceptions.Timeout:
        return pd.DataFrame(), "OpenSky request timed out."
    except Exception as e:
        return pd.DataFrame(), f"OpenSky fetch error: {type(e).__name__}: {str(e)[:200]}"