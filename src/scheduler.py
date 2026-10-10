"""
scheduler.py — Main entry point for the Surf Crowd Monitor pipeline.

Runs continuously, sampling all enabled spots every 15 minutes during
the active window (1hr before sunrise to 1hr after sunset per spot).

Usage:
  python src/scheduler.py           # continuous loop
  python src/scheduler.py --once    # single sample cycle then exit (for testing)
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

import anthropic
from astral import LocationInfo
from astral.sun import sun
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root (no-op if file absent)

# ── Path setup so sibling modules import cleanly ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from capture    import fetch_burst, score_frames
from conditions import fetch_conditions
from db         import (get_client, write_observation, upload_frame, recent_observations,
                        save_regional_report)
from detect     import analyze_frame

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config paths ──────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
_SPOTS    = _ROOT / "config" / "spots.json"
_SETTINGS = _ROOT / "config" / "settings.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_config():
    spots    = json.loads(_SPOTS.read_text())["spots"]
    settings = json.loads(_SETTINGS.read_text())
    return spots, settings


def active_window(spot: dict, dt_utc: datetime) -> tuple[datetime, datetime]:
    """Return (window_start, window_end) in the spot's local timezone."""
    from zoneinfo import ZoneInfo
    tz       = ZoneInfo(spot["timezone"])
    dt_local = dt_utc.astimezone(tz)
    loc      = LocationInfo(
        name      = spot["name"],
        region    = "US",
        timezone  = spot["timezone"],
        latitude  = spot["lat"],
        longitude = spot["lng"],
    )
    s     = sun(loc.observer, date=dt_local.date(), tzinfo=tz)
    start = s["sunrise"] - timedelta(hours=1)
    end   = s["sunset"]  + timedelta(hours=1)
    return start, end


def is_active(spot: dict, dt_utc: datetime) -> bool:
    from zoneinfo import ZoneInfo
    tz       = ZoneInfo(spot["timezone"])
    dt_local = dt_utc.astimezone(tz)
    start, end = active_window(spot, dt_utc)
    return start <= dt_local <= end


def seconds_until_next_interval(interval_minutes: int) -> float:
    """Seconds to sleep until the next clean interval boundary (e.g. :00, :15, :30, :45)."""
    now             = datetime.now(timezone.utc)
    total_secs      = now.minute * 60 + now.second + now.microsecond / 1e6
    interval_secs   = interval_minutes * 60
    remainder       = total_secs % interval_secs
    sleep_secs      = interval_secs - remainder
    return max(sleep_secs, 1.0)   # at least 1s to avoid tight loops


def next_active_start(spots: list[dict], dt_utc: datetime):
    """
    Find the earliest window_start across all enabled spots that is in the future.
    Returns None if no future window can be determined today.
    """
    candidates = []
    for spot in spots:
        if not spot.get("enabled"):
            continue
        try:
            start, _ = active_window(spot, dt_utc)
            if start > dt_utc.astimezone(start.tzinfo):
                candidates.append(start.astimezone(timezone.utc))
        except Exception:
            pass
    return min(candidates) if candidates else None


# ─── Cost-saving plan per spot ────────────────────────────────────────────────

_COUNT_METHOD  = "claude_vision_full_frame_best_frame_v8_points"
_COUNT_ONLY    = "_count_only"
_VISION_FIELDS = [
    "vision_surface", "vision_swell_size", "vision_wave_quality", "vision_wind_effect",
    "vision_crowd_distribution", "vision_water_clarity", "vision_lighting",
    "vision_visibility", "vision_conditions_notes",
]


def plan_spot(rows: list[dict], now: datetime, settings: dict) -> tuple[str, dict | None]:
    """
    Decide what this tick does for one spot, from its recent rows (newest first):
      ("skip",  None)  quiet lineup sampled recently — sample every 30 min, not 15
      ("count", row)   conditions assessed within the hour — count only, carry `row`'s
      ("full",  None)  count + conditions assessment
    Each threshold has 5 minutes of slack so a late GitHub Actions tick still lines up.
    """
    def age_min(row):
        return (now - datetime.fromisoformat(row["captured_at"])).total_seconds() / 60

    if rows:
        last = rows[0]
        if (last.get("count_reliable")
                and last.get("surfer_count") is not None
                and last["surfer_count"] <= settings["quiet_count_threshold"]
                and age_min(last) < settings["quiet_interval_minutes"] - 5):
            return "skip", None

    for row in rows:
        if (not (row.get("count_method") or "").endswith(_COUNT_ONLY)
                and row.get("vision_surface")
                and age_min(row) < settings["conditions_interval_minutes"] - 5):
            return "count", row

    return "full", None


# ─── Sample cycle ─────────────────────────────────────────────────────────────

def run_spot_sample(
    spot:         dict,
    settings:     dict,
    claude_client: anthropic.Anthropic,
    output_dir:   Path,
    skip_db:      bool = False,
    last_assessment: dict | None = None,
):
    """
    Run a full sample cycle for one spot. When `last_assessment` (a recent row
    with a conditions assessment) is given, Claude only counts and that row's
    vision_* conditions are carried into this record.

      1. Fetch burst
      2. Score quality
      3. Claude vision count on qualifying frames
      4. Fetch Surfline conditions
      5. Write to Supabase (unless skip_db=True)

    Returns an outcome string: "ok", "no_frames" (transient capture problem),
    or "write_failed" (the sample was lost).
    """
    spot_id    = spot["id"]
    captured_at = datetime.now(timezone.utc)

    log.info(f"━━━ [{spot_id}] Sample at {captured_at.strftime('%Y-%m-%d %H:%M:%S')} UTC ━━━")

    # 1. Fetch burst
    frame_paths = fetch_burst(spot, settings, output_dir)
    if not frame_paths:
        log.warning(f"[{spot_id}] No frames fetched — skipping")
        return "no_frames"

    # 2. Score quality
    scored = score_frames(frame_paths, settings)

    # 3. Claude vision on the best qualifying frame only (cost optimisation)
    for fr in scored:
        fr.update({"count": -1, "confidence": "n/a", "notes": "excluded: low quality"})

    qualifying = [fr for fr in scored if fr["quality"]["contributes"]]
    if qualifying:
        best_qual = max(qualifying, key=lambda x: x["quality"]["overall_score"])
        result = analyze_frame(
            best_qual["image"], f"{spot_id}/F{best_qual['index']}", claude_client, settings,
            include_conditions=last_assessment is None,
        )
        best_qual.update(result)
        log.info(
            f"[{spot_id}] Sent 1 frame to Claude "
            f"(best quality: F{best_qual['index']} score={best_qual['quality']['overall_score']:.2f})"
        )

    # 4. Determine final count
    contributing = [
        fr for fr in scored
        if fr.get("count", -1) >= 0 and fr["quality"]["contributes"]
    ]

    if contributing:
        best             = contributing[0]  # only one frame was sent to Claude
        surfer_count     = best["count"]
        count_reliable   = True
        claude_notes     = best["notes"]
        vision_conditions      = best.get("conditions", {})
        vision_conditions_notes = best.get("conditions_notes", "")
        log.info(
            f"[{spot_id}] Final count: {surfer_count}  "
            f"(best-frame, 1 API call)"
        )
    else:
        surfer_count     = None   # null = unknown; 0 would mean "counted zero"
        count_reliable   = False
        claude_notes     = "No qualifying frames"
        vision_conditions      = {}
        vision_conditions_notes = ""
        log.warning(f"[{spot_id}] No qualifying frames — count unreliable")

    # 5. Quality averages
    all_scores  = [fr["quality"]["overall_score"] for fr in scored]
    all_laps    = [fr["quality"]["lap_var"]       for fr in scored]
    all_noise   = [fr["quality"]["noisy_pct"]     for fr in scored]

    # 5b. Pick best frame image for storage (highest quality score)
    best_frame_img = max(scored, key=lambda x: x["quality"]["overall_score"])["image"]

    # 6. Surfline conditions
    conditions = fetch_conditions(spot, captured_at)

    # 7. Build record
    record = {
        "captured_at":      captured_at.isoformat(),
        "spot_id":          spot_id,
        "spot_name":        spot["name"],
        "surfer_count":     surfer_count,
        "count_reliable":   count_reliable,
        "count_method":     _COUNT_METHOD + ("" if last_assessment is None else _COUNT_ONLY),
        "session_quality":  round(mean(all_scores), 3),
        "frame_quality_avg": round(mean(all_scores), 3),
        "lap_var_avg":      round(mean(all_laps), 1),
        "noisy_pct_avg":    round(mean(all_noise), 1),
        "wave_height_min":  conditions["wave_height_min"],
        "wave_height_max":  conditions["wave_height_max"],
        "swell_height":     conditions["swell_height"],
        "swell_period":     conditions["swell_period"],
        "swell_direction":  conditions["swell_direction"],
        "wind_speed":       conditions["wind_speed"],
        "wind_direction":   conditions["wind_direction"],
        "tide_height":      conditions["tide_height"],
        "spot_rating":      conditions["spot_rating"],
        "conditions_raw":   conditions["conditions_raw"],
        "vision_surface":            vision_conditions.get("surface"),
        "vision_swell_size":         vision_conditions.get("swell_size"),
        "vision_wave_quality":       vision_conditions.get("wave_quality"),
        "vision_wind_effect":        vision_conditions.get("wind_effect"),
        "vision_crowd_distribution": vision_conditions.get("crowd_distribution"),
        "vision_water_clarity":      vision_conditions.get("water_clarity"),
        "vision_lighting":           vision_conditions.get("lighting"),
        "vision_visibility":         vision_conditions.get("visibility"),
        "vision_conditions_notes":   vision_conditions_notes,
        "frames_raw": [
            {
                "index":      fr["index"],
                "count":      fr.get("count", -1),
                "confidence": fr.get("confidence", "n/a"),
                "notes":      fr.get("notes", ""),
                **{k: fr[k] for k in ("model", "input_tokens", "output_tokens", "points") if k in fr},
                "quality": {
                    "score":     fr["quality"]["overall_score"],
                    "grade":     fr["quality"]["grade"],
                    "lap_var":   fr["quality"]["lap_var"],
                    "noisy_pct": fr["quality"]["noisy_pct"],
                },
            }
            for fr in scored
        ],
        "claude_notes": claude_notes,
    }

    if last_assessment is not None:
        # Conditions barely move in an hour; reuse the latest assessment so the
        # dashboard keeps showing them. count_method marks the row as count-only.
        record.update({k: last_assessment.get(k) for k in _VISION_FIELDS})

    # 8. Upload best frame to Supabase Storage
    frame_url = None
    if not skip_db:
        frame_url = upload_frame(spot_id, captured_at, best_frame_img)
    record["frame_url"] = frame_url

    # 9. Write to DB
    if not skip_db:
        if write_observation(record) is None:
            # Do not let a failed write pass for success: an unnoticed DB
            # outage silently dropped eight days of samples in Aug 2026.
            log.error(f"[{spot_id}] observation was NOT persisted")
            return "write_failed"
        if conditions["report"]:
            save_regional_report(conditions["report"])
    else:
        log.info(f"[{spot_id}] skip_db=True — not writing to Supabase")

    return "ok"


def run_sample_cycle(
    spots:         list[dict],
    settings:      dict,
    claude_client: anthropic.Anthropic,
    output_dir:    Path,
    skip_db:       bool = False,
) -> dict[str, int]:
    """
    Run one full sample cycle across all enabled spots sequentially.

    Returns a tally of outcomes so the caller can decide whether the cycle
    was healthy enough to report success.
    """
    enabled = [s for s in spots if s.get("enabled")]
    log.info(f"Starting sample cycle — {len(enabled)} enabled spot(s)")

    # Recent rows drive the quiet-lineup skip and the hourly conditions
    # assessment. Without them (no DB, or the read failed) every spot gets a
    # full sample, which costs more but never loses data.
    now    = datetime.now(timezone.utc)
    window = max(settings["conditions_interval_minutes"], settings["quiet_interval_minutes"])
    recent = None if skip_db else recent_observations(now - timedelta(minutes=window))
    if recent is None:
        log.warning("No recent observations available — full sample for every spot")

    tally = {"ok": 0, "no_frames": 0, "write_failed": 0, "error": 0, "skipped": 0}
    for spot in enabled:
        try:
            rows = [r for r in (recent or []) if r["spot_id"] == spot["id"]]
            action, last_assessment = plan_spot(rows, now, settings)
            if action == "skip":
                log.info(f"[{spot['id']}] Quiet lineup sampled recently — skipping this tick")
                tally["skipped"] += 1
                continue
            outcome = run_spot_sample(spot, settings, claude_client, output_dir,
                                      skip_db=skip_db, last_assessment=last_assessment)
            tally[outcome] = tally.get(outcome, 0) + 1
        except Exception as e:
            log.error(f"[{spot['id']}] Unhandled error: {e}", exc_info=True)
            tally["error"] += 1

    log.info(
        f"Cycle tally — ok={tally['ok']} no_frames={tally['no_frames']} "
        f"write_failed={tally['write_failed']} error={tally['error']} skipped={tally['skipped']}"
    )
    return tally


# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Surf Crowd Monitor scheduler")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sample cycle immediately then exit (skips DB write if Supabase not configured)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass active window check (useful for testing outside daylight hours)",
    )
    args = parser.parse_args()

    # Validate required env vars
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY is not set")
        sys.exit(1)

    supabase_ready = bool(
        os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")
    )
    if not supabase_ready:
        log.warning("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — DB writes will be skipped")

    spots, settings = load_config()
    claude_client   = anthropic.Anthropic(api_key=api_key)
    output_dir      = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Surf Crowd Monitor starting up")
    log.info(f"Spots: {[s['id'] for s in spots if s.get('enabled')]}")
    log.info(f"Interval: every {settings['sample_interval_minutes']} minutes")
    log.info(f"Active window: sunrise-1h → sunset+1h per spot")

    if args.once:
        log.info("--once flag set — running single cycle")
        now          = datetime.now(timezone.utc)
        enabled      = [s for s in spots if s.get("enabled")]
        if args.force:
            log.info("--force flag set — bypassing active window check")
            active_spots = enabled
        else:
            active_spots = [s for s in enabled if is_active(s, now)]
        if not active_spots:
            log.info("Outside active window for all spots — nothing to do. Use --force to override.")
            log.info("Done.")
            return

        tally = run_sample_cycle(
            active_spots, settings, claude_client, output_dir, skip_db=not supabase_ready
        )

        # Exit non-zero so a scheduled run turns red instead of reporting a
        # green build while dropping data. A single offline cam is tolerated;
        # a failed write, or every spot failing, is not.
        failed = tally["write_failed"] + tally["error"]
        if failed:
            log.error(f"{failed} spot(s) failed to persist — failing the run")
            sys.exit(1)
        if tally["ok"] == 0 and tally["skipped"] == 0:
            log.error("No spot produced a usable sample — failing the run")
            sys.exit(1)

        log.info("Done.")
        return

    # ── Continuous loop ────────────────────────────────────────────────────────
    while True:
        now          = datetime.now(timezone.utc)
        active_spots = [s for s in spots if s.get("enabled") and is_active(s, now)]

        if active_spots:
            log.info(f"Active spots: {[s['id'] for s in active_spots]}")
            # The long-running loop logs the tally and keeps going; it must not
            # exit on a transient failure the way the one-shot CI run does.
            run_sample_cycle(
                active_spots, settings, claude_client, output_dir, skip_db=not supabase_ready
            )
        else:
            next_start = next_active_start(spots, now)
            if next_start:
                sleep_secs = (next_start - now).total_seconds()
                wake_str   = next_start.strftime("%H:%M:%S UTC")
                log.info(f"Outside active window for all spots. Sleeping {sleep_secs/3600:.1f}h until {wake_str}")
                time.sleep(max(sleep_secs - 30, 60))  # wake 30s early to re-evaluate
                continue
            else:
                log.info("Outside active window. Sleeping 30 minutes.")
                time.sleep(1800)
                continue

        sleep_secs = seconds_until_next_interval(settings["sample_interval_minutes"])
        log.info(f"Cycle complete. Next sample in {sleep_secs:.0f}s")
        time.sleep(sleep_secs)


if __name__ == "__main__":
    main()
