"""
conditions.py — Surfline conditions enrichment via Supabase Edge Function.

Calls the /functions/v1/conditions Edge Function which proxies Surfline
server-side. This avoids Cloudflare blocking CI runner IPs when calling
Surfline directly.

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables
(already needed by db.py).
"""

import logging
import os
from datetime import datetime

import requests

log = logging.getLogger(__name__)

_TIMEOUT = 15


def fetch_conditions(spot: dict, captured_at: datetime) -> dict:
    """
    Fetch conditions from the Supabase Edge Function proxy for the given spot.

    Returns a flat dict with all condition fields plus a conditions_raw key.
    Fields are None when unavailable (function error, missing data, wrong ID).
    """
    surfline_id = spot["surfline_spot_id"]
    spot_id     = spot["id"]

    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "")

    if not supabase_url:
        log.warning(f"[{spot_id}] SUPABASE_URL not set — skipping conditions fetch")
        return _empty()

    url = f"{supabase_url}/functions/v1/conditions"

    log.info(f"[{spot_id}] Fetching conditions via Edge Function (spotId={surfline_id})")

    try:
        r = requests.get(
            url,
            params={"spotId": surfline_id},
            headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"[{spot_id}] Conditions Edge Function failed: {e}")
        return _empty()

    log.info(
        f"[{spot_id}] Wave: {data.get('wave_height_min')}–{data.get('wave_height_max')}ft  "
        f"Swell: {data.get('swell_height')}ft @ {data.get('swell_period')}s  "
        f"Wind: {data.get('wind_speed')}kts {data.get('wind_direction_type')}  "
        f"Tide: {data.get('tide_height')}ft  "
        f"Rating: {data.get('spot_rating')}"
    )
    if data.get("upstream_failures"):
        log.warning(f"[{spot_id}] Surfline calls failed after retries: {data['upstream_failures']}")
    if data.get("report"):
        rep = data["report"]
        log.info(f"[{spot_id}] Report: {rep.get('subregion_name')} @ {rep.get('published_at')} "
                 f"by {rep.get('forecaster')} — {str(rep.get('headline'))[:80]}")

    return {
        "wave_height_min":  data.get("wave_height_min"),
        "wave_height_max":  data.get("wave_height_max"),
        "swell_height":     data.get("swell_height"),
        "swell_period":     data.get("swell_period"),
        "swell_direction":  data.get("swell_direction"),
        "wind_speed":       data.get("wind_speed"),
        "wind_direction":   data.get("wind_direction"),
        "tide_height":      data.get("tide_height"),
        "spot_rating":      data.get("spot_rating"),
        # Regional written forecast, stored once per subregion in its own table
        # rather than copied onto every observation's conditions_raw.
        "report":           data.get("report"),
        "conditions_raw":   {k: v for k, v in data.items() if k != "report"},
    }


def _empty() -> dict:
    return {
        "wave_height_min":  None,
        "wave_height_max":  None,
        "swell_height":     None,
        "swell_period":     None,
        "swell_direction":  None,
        "wind_speed":       None,
        "wind_direction":   None,
        "tide_height":      None,
        "spot_rating":      None,
        "report":           None,
        "conditions_raw":   {},
    }
