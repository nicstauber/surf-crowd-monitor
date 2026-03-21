"""
Surf Crowd Monitor — Spike v5
==============================
Replaces YOLO with Claude vision for surfer detection.

Instead of a generic person detector, we send cropped water zone tiles
directly to Claude's vision API and ask it to count surfers specifically.
Claude understands context — sitting on boards, prone paddling, partially
submerged — in a way a generic CV model doesn't.

Approach:
  - Capture 3-frame burst as before
  - Score each frame for quality
  - For qualifying frames: crop the water zone, send to Claude vision API
  - Claude returns a count + brief description of what it sees
  - Max count across burst is the final reported number

Cost estimate: ~3 API calls per spot per interval (one per frame).
At claude-haiku-3-5 pricing this is fractions of a cent per sample.

Requirements:
  pip3 install requests anthropic opencv-python pillow

Usage:
  python3 surf_spike_v5.py                               # fresh burst
  python3 surf_spike_v5.py --local f1.jpg f2.jpg f3.jpg  # existing frames
  python3 surf_spike_v5.py --local spike_output/frame_1.jpg --single  # one frame

Setup:
  export ANTHROPIC_API_KEY=your_key_here
  (get your key at https://console.anthropic.com)
"""

import requests
import subprocess
import sys
import os
import json
import time
import base64
import argparse
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path
from datetime import datetime
from io import BytesIO
import anthropic

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SPOTS = {
    "lower_trestles": "https://hls.cdn-surfline.com/oregon/wc-lowerslefts/playlist.m3u8",
    "hb_pier_south":  "https://hls.cdn-surfline.com/oregon/wc-huntingtonbeachsouthside/playlist.m3u8",
    "hb_cliffs":      "https://hls.cdn-surfline.com/oregon/wc-huntingtoncliffs/playlist.m3u8",
    "newport_56th":   "https://hls.cdn-surfline.com/oregon/wc-fiftysixnewport/playlist.m3u8",
}
TEST_SPOT = "newport_56th"

# Water zone — fraction of frame height (top, bottom)
# Tune per cam. The zone should cover the lineup and exclude beach/sky.
WATER_ZONE = {
    "lower_trestles": (0.28, 0.65),
    "hb_pier_south":  (0.25, 0.65),
    "hb_cliffs":      (0.25, 0.65),
    "newport_56th":   (0.20, 0.55),   # lower cam angle — water is higher in frame
}

# ── Burst config ──────────────────────────────────────────────────────────────
BURST_FRAME_COUNT      = 3
BURST_INTERVAL_SECONDS = 10
BURST_MIN_QUALITY      = 0.45

# ── Claude vision config ──────────────────────────────────────────────────────
# Haiku is fast and cheap — great for structured counting tasks
CLAUDE_MODEL = "claude-haiku-4-5"

# Tiling: split water zone into overlapping tiles so surfers aren't tiny
# Claude handles up to 20 images per request but we send one tile at a time
# for cleaner per-region counting
TILE_COLS    = 3    # fewer, larger tiles work better for Claude than YOLO
TILE_ROWS    = 1    # single row — water zone is wide, not tall
TILE_OVERLAP = 0.15

# ── Quality thresholds (calibrated from real frame data) ─────────────────────
LAP_VAR_CLEAN     = 800
LAP_VAR_BAD       = 4000
PATCH_NOISE_CLEAN = 0.25
PATCH_NOISE_BAD   = 0.70
BRIGHTNESS_MIN    = 60
BRIGHTNESS_MAX    = 200
BRIGHTNESS_IDEAL  = 115

HEADERS = {
    "origin":     "https://www.surfline.com",
    "referer":    "https://www.surfline.com/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "accept":     "*/*",
}

OUTPUT_DIR = Path("./spike_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── STREAM FETCH ─────────────────────────────────────────────────────────────

def fetch_single_frame(spot_key, frame_index):
    url = SPOTS[spot_key]
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return None
    lines = [l.strip() for l in r.text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    seg_url = lines[-1] if lines[-1].startswith("http") else f"{url.rsplit('/',1)[0]}/{lines[-1]}"
    r2 = requests.get(seg_url, headers=HEADERS, timeout=30)
    if r2.status_code != 200:
        return None
    seg_path   = OUTPUT_DIR / f"segment_{frame_index}.ts"
    frame_path = OUTPUT_DIR / f"frame_{frame_index}.jpg"
    seg_path.write_bytes(r2.content)
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(seg_path), "-frames:v", "1", "-q:v", "2", str(frame_path)],
        capture_output=True
    )
    return frame_path if res.returncode == 0 else None


def fetch_burst(spot_key):
    print(f"\n[FETCH] Burst: {BURST_FRAME_COUNT} frames × {BURST_INTERVAL_SECONDS}s")
    frames = []
    for i in range(BURST_FRAME_COUNT):
        if i > 0:
            print(f"  Waiting {BURST_INTERVAL_SECONDS}s..."); time.sleep(BURST_INTERVAL_SECONDS)
        print(f"  Frame {i+1}/{BURST_FRAME_COUNT}...", end=" ", flush=True)
        path = fetch_single_frame(spot_key, i + 1)
        if path:
            print("✓"); frames.append(path)
        else:
            print("✗")
    return frames

# ─── FRAME QUALITY SCORER ─────────────────────────────────────────────────────

def score_frame_quality(img_pil, spot_key, frame_label=""):
    w, h = img_pil.size
    zone_top, zone_bot = WATER_ZONE.get(spot_key, (0.25, 0.65))
    wy1 = int(h * zone_top); wy2 = int(h * zone_bot)
    water_np   = np.array(img_pil.crop((0, wy1, w, wy2)))
    water_gray = cv2.cvtColor(water_np, cv2.COLOR_RGB2GRAY)

    lap     = cv2.Laplacian(water_gray, cv2.CV_64F)
    lap_var = float(lap.var())
    lap_score = 1.0 - min(1.0, max(0.0, (lap_var - LAP_VAR_CLEAN) / (LAP_VAR_BAD - LAP_VAR_CLEAN)))

    patch_stds  = [float(water_gray[y:y+16, x:x+16].std())
                   for y in range(0, water_gray.shape[0]-16, 16)
                   for x in range(0, water_gray.shape[1]-16, 16)]
    noisy_frac  = float(np.mean(np.array(patch_stds) > 20))
    patch_score = 1.0 - min(1.0, max(0.0, (noisy_frac - PATCH_NOISE_CLEAN) / (PATCH_NOISE_BAD - PATCH_NOISE_CLEAN)))
    glare_score = (lap_score + patch_score) / 2.0

    mean_b = float(water_gray.mean())
    if mean_b < BRIGHTNESS_MIN:
        b_score = mean_b / BRIGHTNESS_MIN
    elif mean_b > BRIGHTNESS_MAX:
        b_score = max(0.0, 1.0 - (mean_b - BRIGHTNESS_MAX) / 55.0)
    else:
        b_score = 1.0 - abs(mean_b - BRIGHTNESS_IDEAL) / 80.0
    b_score = max(0.0, min(1.0, b_score))

    std = float(water_gray.std())
    c_score = min(1.0, std / 25.0) if std < 25 else 1.0

    overall = round(glare_score * 0.55 + b_score * 0.25 + c_score * 0.20, 3)

    if overall >= 0.75:   grade = "GOOD"
    elif overall >= 0.62: grade = "MARGINAL"
    elif overall >= 0.45: grade = "POOR"
    else:                 grade = "UNUSABLE"

    return {
        "overall_score":   overall,
        "grade":           grade,
        "reliable":        overall >= 0.62,
        "contributes":     overall >= BURST_MIN_QUALITY,
        "lap_var":         round(lap_var, 1),
        "noisy_pct":       round(noisy_frac * 100, 1),
        "mean_brightness": round(mean_b, 1),
        "water_zone":      (wy1, wy2),
    }

# ─── CLAUDE VISION COUNTING ───────────────────────────────────────────────────

def pil_to_base64(img_pil, max_width=1024):
    """Resize if needed and encode as base64 JPEG."""
    if img_pil.width > max_width:
        ratio = max_width / img_pil.width
        img_pil = img_pil.resize(
            (max_width, int(img_pil.height * ratio)), Image.LANCZOS
        )
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def count_surfers_claude(water_zone_img, tile_label, client):
    """
    Send a water zone tile to Claude vision and ask for a surfer count.
    Returns (count, description, raw_response).
    """
    b64 = pil_to_base64(water_zone_img)

    prompt = """You are analyzing a surf cam image to count surfers in the water.

Look carefully at this cropped section of a surf cam image. Count every person who is:
- Sitting on a surfboard waiting for waves (in the lineup)
- Lying on a surfboard paddling
- Standing on a wave / actively surfing
- Visible in the water in any way, even partially

Do NOT count:
- People on the beach
- People on the pier or rocks
- Birds or other animals
- Unclear blobs that might not be people

Respond with ONLY a JSON object in this exact format:
{
  "count": <integer>,
  "confidence": "<low|medium|high>",
  "notes": "<brief description of what you see, e.g. '3 surfers in lineup, 1 on wave'>"
}

If the image is too dark, glary, or unclear to make a reasonable count, set count to -1 and confidence to "low".
Respond with ONLY the JSON. No other text."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       b64,
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        count  = int(parsed.get("count", -1))
        conf   = parsed.get("confidence", "unknown")
        notes  = parsed.get("notes", "")
        return count, conf, notes, raw

    except json.JSONDecodeError as e:
        print(f"    ⚠  JSON parse error on {tile_label}: {e}\n    Raw: {raw[:200]}")
        return -1, "error", f"JSON parse error: {e}", ""
    except Exception as e:
        print(f"    ⚠  API error on {tile_label}: {e}")
        return -1, "error", str(e), ""


def detect_with_claude(img_pil, spot_key, frame_label, client):
    """
    Tile the water zone and count surfers in each tile using Claude vision.
    Returns total count (deduplicated) and per-tile detail.
    """
    w, h = img_pil.size
    zone_top, zone_bot = WATER_ZONE.get(spot_key, (0.25, 0.65))
    wy1 = int(h * zone_top)
    wy2 = int(h * zone_bot)

    water = img_pil.crop((0, wy1, w, wy2))
    ww, wh = water.size

    tile_w = int(ww / (TILE_COLS - TILE_OVERLAP * (TILE_COLS - 1)))
    step_x = int(tile_w * (1 - TILE_OVERLAP))

    tile_results = []
    for col in range(TILE_COLS):
        tx1 = col * step_x
        tx2 = min(tx1 + tile_w, ww)
        tile = water.crop((tx1, 0, tx2, wh))
        label = f"{frame_label}_T{col+1}"

        print(f"    Tile {col+1}/{TILE_COLS}...", end=" ", flush=True)
        count, conf, notes, raw = count_surfers_claude(tile, label, client)
        print(f"count={count}  conf={conf}  → {notes}")

        tile_results.append({
            "tile":     col + 1,
            "x_range":  (tx1, tx2),
            "count":    count,
            "conf":     conf,
            "notes":    notes,
        })

    # Sum valid tile counts
    # Surfers in the overlap zone between tiles could be double-counted.
    # We sum counts but apply a small dedup discount for tiles with overlap.
    valid_counts = [t["count"] for t in tile_results if t["count"] >= 0]
    if not valid_counts:
        total = -1
    elif TILE_COLS == 1:
        total = valid_counts[0]
    else:
        # Simple overlap correction: subtract estimated overlap count
        # Each border between tiles shares ~15% overlap
        # Conservative: assume at most 1 person per overlap boundary
        raw_sum = sum(valid_counts)
        overlap_deduction = (TILE_COLS - 1) * 1  # conservative
        total = max(0, raw_sum - overlap_deduction)

    return total, tile_results, (wy1, wy2)

# ─── ANNOTATED CONTACT SHEET ──────────────────────────────────────────────────

def save_contact_sheet(frame_results, final_count, session_quality, ts, spot_key):
    if not frame_results:
        return None

    annotated = []
    for fr in frame_results:
        img_np = cv2.cvtColor(np.array(fr["image"]), cv2.COLOR_RGB2BGR)
        wy1, wy2 = fr["quality"]["water_zone"]
        w = img_np.shape[1]

        # Water zone box
        cv2.rectangle(img_np, (0, wy1), (w, wy2), (0, 165, 255), 2)

        # Tile dividers
        ww = w
        wh = wy2 - wy1
        tile_w = int(ww / (TILE_COLS - TILE_OVERLAP * (TILE_COLS - 1)))
        step_x = int(tile_w * (1 - TILE_OVERLAP))
        for col in range(1, TILE_COLS):
            tx = col * step_x
            cv2.line(img_np, (tx, wy1), (tx, wy2), (0, 200, 200), 1)

        # Per-tile counts
        for t in fr.get("tiles", []):
            tx1 = t["x_range"][0]
            label = f"T{t['tile']}: {t['count']} ({t['conf'][0].upper()})"
            cv2.putText(img_np, label, (tx1 + 8, wy1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)
            if t["notes"]:
                cv2.putText(img_np, t["notes"][:40], (tx1 + 8, wy1 + 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1)

        # Frame badge
        grade   = fr["quality"]["grade"]
        score   = fr["quality"]["overall_score"]
        count   = fr["count"]
        q_color = (0,200,0) if grade=="GOOD" else (0,165,255) if grade=="MARGINAL" else (0,0,220)
        is_max  = fr.get("is_max", False)

        cv2.rectangle(img_np, (0, 0), (620, 95), (0, 0, 0), -1)
        cv2.putText(img_np,
                    f"{'★ MAX  ' if is_max else ''}Frame {fr['index']}  |  {ts.strftime('%H:%M:%S')} UTC",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 100) if is_max else (200, 200, 200), 1)
        cv2.putText(img_np, f"SURFERS: {count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 100) if is_max else (200, 200, 200), 2)
        cv2.putText(img_np, f"Quality: {score:.2f} [{grade}]  Lap={fr['quality']['lap_var']:.0f}",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.58, q_color, 1)

        scale = 640 / img_np.shape[1]
        small = cv2.resize(img_np, (640, int(img_np.shape[0] * scale)))
        annotated.append(small)

    max_h = max(a.shape[0] for a in annotated)
    padded = [cv2.copyMakeBorder(a, 0, max_h - a.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(20, 20, 20))
              for a in annotated]
    sheet = np.hstack(padded)

    bar_h   = 70
    bar     = np.zeros((bar_h, sheet.shape[1], 3), dtype=np.uint8)
    q_color = (0,200,0) if session_quality >= 0.75 else \
              (0,165,255) if session_quality >= 0.62 else (0,0,220)
    cv2.putText(bar,
                f"FINAL COUNT: {final_count}  (max across {len(frame_results)} frames)   |   "
                f"Session quality: {session_quality:.2f}   |   Model: {CLAUDE_MODEL}",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, q_color, 2)

    final_img = np.vstack([sheet, bar])
    out_path  = OUTPUT_DIR / "burst_contact_sheet.jpg"
    cv2.imwrite(str(out_path), final_img)
    return out_path

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", nargs="+", help="Paths to existing frame jpgs")
    parser.add_argument("--single", action="store_true", help="Run on first frame only (quick test)")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗  ANTHROPIC_API_KEY not set.")
        print("   Get your key at https://console.anthropic.com")
        print("   Then run:  export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    now    = datetime.utcnow()

    print("=" * 62)
    print("SURF CROWD MONITOR — v5  (Claude vision counting)")
    print(f"Spot:   {TEST_SPOT}")
    print(f"Time:   {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Model:  {CLAUDE_MODEL}")
    print(f"Burst:  {BURST_FRAME_COUNT} frames × {BURST_INTERVAL_SECONDS}s")
    print("=" * 62)

    # ── 1. Get frames ─────────────────────────────────────────────────────
    if args.local:
        frame_paths = [Path(p) for p in args.local]
        if args.single:
            frame_paths = frame_paths[:1]
        print(f"\n[FETCH] Using {len(frame_paths)} local frames")
    else:
        frame_paths = fetch_burst(TEST_SPOT)
        if args.single:
            frame_paths = frame_paths[:1]

    if not frame_paths:
        print("✗ No frames"); sys.exit(1)

    # ── 2. Score quality + count with Claude ──────────────────────────────
    print(f"\n[PROCESS] Scoring + counting {len(frame_paths)} frames...")
    frame_results = []

    for i, fp in enumerate(frame_paths):
        label   = f"F{i+1}"
        img_pil = Image.open(fp).convert("RGB")
        quality = score_frame_quality(img_pil, TEST_SPOT, label)

        print(f"\n  {label}: quality={quality['overall_score']:.2f} [{quality['grade']}]  "
              f"Lap={quality['lap_var']:.0f}  Noise={quality['noisy_pct']:.0f}%")

        if not quality["contributes"]:
            print(f"  → Below threshold {BURST_MIN_QUALITY} — EXCLUDED")
            frame_results.append({
                "index": i+1, "path": str(fp), "image": img_pil,
                "quality": quality, "tiles": [], "count": -1
            })
            continue

        print(f"  → Sending to Claude vision ({TILE_COLS} tiles)...")
        total, tiles, water_zone_y = detect_with_claude(img_pil, TEST_SPOT, label, client)
        quality["water_zone"] = water_zone_y  # update with actual zone used

        print(f"  → Frame total: {total} surfers")
        frame_results.append({
            "index": i+1, "path": str(fp), "image": img_pil,
            "quality": quality, "tiles": tiles, "count": total
        })

    # ── 3. Final count ────────────────────────────────────────────────────
    contributing = [fr for fr in frame_results
                    if fr["count"] >= 0 and fr["quality"]["contributes"]]

    if not contributing:
        final_count    = 0
        count_reliable = False
        print("\n⚠  No qualifying frames — count unreliable")
    else:
        best          = max(contributing, key=lambda x: x["count"])
        final_count   = best["count"]
        count_reliable = True
        best["is_max"] = True
        print(f"\n  Final count: {final_count}  (max from Frame {best['index']})")
        counts = [fr["count"] for fr in frame_results]
        print(f"  Per-frame counts: {counts}")

    session_quality = round(
        sum(fr["quality"]["overall_score"] for fr in contributing) / len(contributing), 3
    ) if contributing else 0.0

    # ── 4. Contact sheet ──────────────────────────────────────────────────
    sheet_path = save_contact_sheet(frame_results, final_count, session_quality, now, TEST_SPOT)
    if sheet_path:
        print(f"\n  Contact sheet → {sheet_path}")

    # ── 5. DB record ──────────────────────────────────────────────────────
    record = {
        "spot":            TEST_SPOT,
        "timestamp":       now.isoformat(),
        "surfer_count":    final_count,
        "count_reliable":  count_reliable,
        "count_method":    "claude_vision_max_burst",
        "model":           CLAUDE_MODEL,
        "burst_config": {
            "frame_count":      BURST_FRAME_COUNT,
            "interval_seconds": BURST_INTERVAL_SECONDS,
            "min_quality":      BURST_MIN_QUALITY,
            "tile_cols":        TILE_COLS,
        },
        "session_quality": session_quality,
        "frames": [
            {
                "index":     fr["index"],
                "count":     fr["count"],
                "is_max":    fr.get("is_max", False),
                "quality":   {
                    "score":     fr["quality"]["overall_score"],
                    "grade":     fr["quality"]["grade"],
                    "lap_var":   fr["quality"]["lap_var"],
                    "noisy_pct": fr["quality"]["noisy_pct"],
                },
                "tiles": fr.get("tiles", [])
            }
            for fr in frame_results
        ]
    }

    print("\n" + "=" * 62)
    print("DB RECORD")
    print("=" * 62)
    print(json.dumps(record, indent=2))
    print("=" * 62)
