"""
db.py — Supabase read/write for surf observations.

Requires environment variables:
  SUPABASE_URL         — your project URL (https://xxxx.supabase.co)
  SUPABASE_SERVICE_KEY — service role key (bypasses RLS)
"""

import logging
import os
from io import BytesIO
from datetime import datetime

from PIL import Image
from supabase import create_client, Client

log = logging.getLogger(__name__)

_TABLE = "observations"
_client = None  # type: Client


def get_client() -> Client:
    """Return a cached Supabase client, initialised from environment variables."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment."
            )
        _client = create_client(url, key)
        log.info("Supabase client initialised")
    return _client


def write_observation(record: dict):
    """
    Insert one observation record into the observations table.
    Returns the new row's UUID on success, None on failure.
    """
    client = get_client()
    try:
        response = client.table(_TABLE).insert(record).execute()
        row_id   = response.data[0]["id"] if response.data else None
        log.info(f"DB write OK → id={row_id}")
        return row_id
    except Exception as e:
        log.error(f"DB write failed: {e}")
        return None


def upload_frame(spot_id: str, captured_at: datetime, img_pil: Image.Image) -> str | None:
    """
    Upload the best frame JPEG to Supabase Storage bucket 'frames'.
    Returns the public URL, or None on failure.
    Path: frames/{spot_id}/{YYYY-MM-DDTHH-MM-SS}.jpg
    """
    client = get_client()
    ts     = captured_at.strftime("%Y-%m-%dT%H-%M-%S")
    path   = f"{spot_id}/{ts}.jpg"

    # Resize to thumbnail width for storage efficiency
    thumb = img_pil.copy()
    if thumb.width > 640:
        ratio = 640 / thumb.width
        thumb = thumb.resize((640, int(thumb.height * ratio)), Image.LANCZOS)

    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=75)
    buf.seek(0)

    try:
        client.storage.from_("frames").upload(
            path=path,
            file=buf.getvalue(),
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
        url = client.storage.from_("frames").get_public_url(path)
        log.info(f"Frame uploaded → {url}")
        return url
    except Exception as e:
        log.warning(f"Frame upload failed: {e}")
        return None


def latest_observations(spot_id: str, limit: int = 10):
    """Fetch the most recent observations for a spot (for debugging/inspection)."""
    client = get_client()
    try:
        response = (
            client.table(_TABLE)
            .select("*")
            .eq("spot_id", spot_id)
            .order("captured_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:
        log.error(f"DB read failed: {e}")
        return []


def recent_observations(since: datetime) -> list[dict] | None:
    """
    Fetch every spot's observations captured at or after `since`, newest first.
    Returns None (not []) on failure so callers can tell "no rows" from "unknown".
    """
    client = get_client()
    try:
        response = (
            client.table(_TABLE)
            .select("*")
            .gte("captured_at", since.isoformat())
            .order("captured_at", desc=True)
            .execute()
        )
        return response.data or []
    except Exception as e:
        log.error(f"DB read failed: {e}")
        return None
