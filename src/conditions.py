"""
conditions.py — Surfline conditions enrichment.

Fetches wave, wind, tide, and rating data from the unofficial Surfline v2 API
for a given spot at a given timestamp. All endpoints are unauthenticated.

Surfline spot IDs in config/spots.json should be verified against the Surfline
website — find them in the URL when viewing a spot forecast page.
"""

import logging
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)

_BASE    = "https://services.surfline.com/kbyg/spots/forecasts"
_HEADERS = {
    "origin":     "https://www.surfline.com",
    "referer":    "https://www.surfline.com/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
_TIMEOUT = 10


def _get(endpoint: str, spot_id: str, days: int = 1, interval_hours: int = 1):
    """Fetch one forecast endpoint. Returns parsed JSON or None on failure."""
    params = {"spotId": spot_id, "days": days, "intervalHours": interval_hours}
    try:
        r = requests.get(
            f"{_BASE}/{endpoint}",
            params=params,
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Surfline {endpoint} fetch failed for {spot_id}: {e}")
        return None


def _closest(entries: list, ts: float):
    """Return the entry whose 'timestamp' key is closest to ts."""
    if not entries:
        return None
    return min(entries, key=lambda e: abs(e.get("timestamp", 0) - ts))


def fetch_conditions(spot: dict, captured_at: datetime) -> dict:
    """
    Fetch conditions from Surfline for the given spot and timestamp.

    Returns a flat dict with all condition fields plus a conditions_raw
    key containing the full raw API responses. Fields are None when
    unavailable (API failure, missing data, wrong spot ID).
    """
    spot_id = spot["surfline_spot_id"]
    ts      = captured_at.timestamp()

    log.info(f"[{spot['id']}] Fetching Surfline conditions (spot_id={spot_id})")

    wave_data   = _get("wave",   spot_id)
    wind_data   = _get("wind",   spot_id)
    tides_data  = _get("tides",  spot_id, interval_hours=1)
    rating_data = _get("rating", spot_id)

    # ── Wave ──────────────────────────────────────────────────────────────────
    wave_height_min = wave_height_max = None
    swell_height = swell_period = swell_direction = None

    if wave_data:
        entry = _closest(wave_data.get("data", {}).get("wave", []), ts)
        if entry:
            surf = entry.get("surf", {})
            wave_height_min = surf.get("min")
            wave_height_max = surf.get("max")
            swells = entry.get("swells", [])
            if swells:
                # dominant swell = first (sorted by impact/height by Surfline)
                dom = swells[0]
                swell_height    = dom.get("height")
                swell_period    = dom.get("period")
                swell_direction = dom.get("direction")
            log.info(
                f"[{spot['id']}] Wave: {wave_height_min}–{wave_height_max}ft  "
                f"Swell: {swell_height}ft @ {swell_period}s {swell_direction}°"
            )

    # ── Wind ──────────────────────────────────────────────────────────────────
    wind_speed = wind_direction = None

    if wind_data:
        entry = _closest(wind_data.get("data", {}).get("wind", []), ts)
        if entry:
            wind_speed     = entry.get("speed")
            wind_direction = entry.get("direction")
            log.info(f"[{spot['id']}] Wind: {wind_speed}kts @ {wind_direction}°")

    # ── Tides ─────────────────────────────────────────────────────────────────
    tide_height = None

    if tides_data:
        entry = _closest(tides_data.get("data", {}).get("tides", []), ts)
        if entry:
            tide_height = entry.get("height")
            log.info(f"[{spot['id']}] Tide: {tide_height}ft")

    # ── Rating ────────────────────────────────────────────────────────────────
    spot_rating = None

    if rating_data:
        entry = _closest(rating_data.get("data", {}).get("rating", []), ts)
        if entry:
            rating_obj = entry.get("rating", {})
            spot_rating = rating_obj.get("key") or rating_obj.get("value")
            log.info(f"[{spot['id']}] Rating: {spot_rating}")

    return {
        "wave_height_min":  wave_height_min,
        "wave_height_max":  wave_height_max,
        "swell_height":     swell_height,
        "swell_period":     swell_period,
        "swell_direction":  swell_direction,
        "wind_speed":       wind_speed,
        "wind_direction":   wind_direction,
        "tide_height":      tide_height,
        "spot_rating":      str(spot_rating) if spot_rating is not None else None,
        "conditions_raw": {
            "wave":   wave_data,
            "wind":   wind_data,
            "tides":  tides_data,
            "rating": rating_data,
        },
    }
