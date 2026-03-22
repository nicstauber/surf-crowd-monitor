"""
Surf Crowd Monitor — Spike v6
==============================
Removes manual water zone detection zones entirely.
Claude vision receives the full frame and uses its own understanding
of the scene to count only surfers in the water — ignoring beach,
pier, rocks, and any other non-water areas automatically.

Changes from v5:
  - No WATER_ZONE config
  - No tiling — full frame sent as single image
  - No manual crop coordinates per cam
  - Claude handles scene understanding natively
  - Works correctly regardless of cam angle

Requirements:
  pip3 install requests anthropic opencv-python pillow

Setup:
  export ANTHROPIC_API_KEY=your_key_here

Usage:
  python3 surf_spike_v6.py                               # fresh burst
  python3 surf_spike_v6.py --local f1.jpg f2.jpg f3.jpg  # existing frames
  python3 surf_spike_v6.py --local spike_output/frame_1.jpg --single
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
from PIL import Image
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

# ── Burst config ──────────────────────────────────────────────────────────────
BURST_FRAME_COUNT      = 3
BURST_INTERVAL_SECONDS = 10
BURST_MIN_QUALITY      = 0.45   # frames below this are excluded from count

# ── Claude vision config ──────────────────────────────────────────────────────
CLAUDE_MODEL       = "claude-haiku-4-5"
CLAUDE_MAX_TOKENS  = 400
CLAUDE_IMAGE_WIDTH = 1280   # resize before sending — keeps tokens reasonable

# ── Quality thresholds ────────────────────────────────────────────────────────
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
    lines = [l.strip() for l in r.text.splitlines()
             if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    seg_url = (lines[-1] if lines[-1].startswith("http")
               else f"{url.rsplit('/',1)[0]}/{lines[-1]}")
    r2 = requests.get(seg_url, headers=HEADERS, timeout=30)
    if r2.status_code != 200:
        return None
    seg_path   = OUTPUT_DIR / f"segment_{frame_index}.ts"
    frame_path = OUTPUT_DIR / f"frame_{frame_index}.jpg"
    seg_path.write_bytes(r2.content)
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(seg_path),
         "-frames:v", "1", "-q:v", "2", str(frame_path)],
        capture_output=True
    )
    return frame_path if res.returncode == 0 else None


def fetch_burst(spot_key):
    print(f"\n[FETCH] Burst: {BURST_FRAME_COUNT} frames × {BURST_INTERVAL_SECONDS}s")
    frames = []
    for i in range(BURST_FRAME_COUNT):
        if i > 0:
            print(f"  Waiting {BURST_INTERVAL_SECONDS}s...", flush=True)
            time.sleep(BURST_INTERVAL_SECONDS)
        print(f"  Frame {i+1}/{BURST_FRAME_COUNT}...", end=" ", flush=True)
        path = fetch_single_frame(spot_key, i + 1)
        if path:
            print("✓"); frames.append(path)
        else:
            print("✗")
    return frames

# ─── FRAME QUALITY SCORER ─────────────────────────────────────────────────────
# Scores the full frame (no water zone crop needed)

def score_frame_quality(img_pil, frame_label=""):
    """Score overall frame quality — used to flag glary/dark frames."""
    # Use the middle vertical third of the frame as proxy for water quality
    # (avoids sky at top and beach at bottom skewing the metrics)
    w, h   = img_pil.size
    mid_y1 = int(h * 0.25)
    mid_y2 = int(h * 0.70)
    sample = np.array(img_pil.crop((0, mid_y1, w, mid_y2)))
    gray   = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)

    # Texture / glare
    lap       = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var   = float(lap.var())
    lap_score = 1.0 - min(1.0, max(0.0,
        (lap_var - LAP_VAR_CLEAN) / (LAP_VAR_BAD - LAP_VAR_CLEAN)))

    patch_stds  = [float(gray[y:y+16, x:x+16].std())
                   for y in range(0, gray.shape[0]-16, 16)
                   for x in range(0, gray.shape[1]-16, 16)]
    noisy_frac  = float(np.mean(np.array(patch_stds) > 20))
    patch_score = 1.0 - min(1.0, max(0.0,
        (noisy_frac - PATCH_NOISE_CLEAN) / (PATCH_NOISE_BAD - PATCH_NOISE_CLEAN)))
    glare_score = (lap_score + patch_score) / 2.0

    # Brightness
    mean_b = float(gray.mean())
    if mean_b < BRIGHTNESS_MIN:
        b_score = mean_b / BRIGHTNESS_MIN
    elif mean_b > BRIGHTNESS_MAX:
        b_score = max(0.0, 1.0 - (mean_b - BRIGHTNESS_MAX) / 55.0)
    else:
        b_score = 1.0 - abs(mean_b - BRIGHTNESS_IDEAL) / 80.0
    b_score = max(0.0, min(1.0, b_score))

    # Contrast
    std     = float(gray.std())
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
    }

# ─── CLAUDE VISION COUNTING ───────────────────────────────────────────────────

def pil_to_base64(img_pil, max_width=CLAUDE_IMAGE_WIDTH):
    """Resize and base64-encode as JPEG."""
    if img_pil.width > max_width:
        ratio   = max_width / img_pil.width
        img_pil = img_pil.resize(
            (max_width, int(img_pil.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def count_surfers_claude(img_pil, frame_label, client):
    """
    Send the full frame to Claude and ask it to count surfers in the water.
    Claude handles scene understanding — no manual crop needed.
    Returns (count, confidence, notes, raw_response).
    """
    b64 = pil_to_base64(img_pil)

    prompt = """You are a precise surf cam analyst. Your job is to count every single person in the water — not estimate, but count each one individually.

COUNTING METHOD — do this mentally before responding:
1. Scan the water from LEFT to RIGHT in horizontal strips
2. In each strip, mark every dark figure, dot, or silhouette on or in the water
3. Count each one — do not round or approximate
4. Move to the next strip and repeat until you've covered the full water area

COUNT THESE — anyone in the ocean, surf zone, or shoreline water:
- Surfers sitting upright on boards in the lineup (usually dark dots/silhouettes)
- Surfers lying prone paddling (elongated shapes on the water surface)
- Surfers actively riding a wave
- Anyone standing, wading, or swimming in the water

DO NOT COUNT:
- People on the dry beach or sand
- People on piers, jetties, rocks, or structures
- Umbrellas, towels, tents, or beach gear
- Birds, animals, or buoys
- Whitewash or foam that resembles a person

IMPORTANT: This is a precise count, not an estimate. If you see 23 people, say 23. If you see 31, say 31. Do not round to the nearest 5 or 10. Count every visible person individually.

Respond ONLY with JSON, no other text:
{
  "count": <exact integer — every person you counted>,
  "confidence": "<low|medium|high>",
  "notes": "<describe where the surfers are, e.g. 'tight cluster of 8 in lineup left, 6 spread across middle, 4 far right near rocks'>"
}

If truly unable to count due to darkness or glare, set count to -1."""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
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
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        raw   = response.content[0].text.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        return (
            int(parsed.get("count", -1)),
            parsed.get("confidence", "unknown"),
            parsed.get("notes", ""),
            raw
        )

    except json.JSONDecodeError as e:
        print(f"    ⚠  JSON parse error: {e}  Raw: {raw[:150]}")
        return -1, "error", f"parse error: {e}", ""
    except Exception as e:
        print(f"    ⚠  API error: {e}")
        return -1, "error", str(e), ""

# ─── CONTACT SHEET ───────────────────────────────────────────────────────────

def save_contact_sheet(frame_results, final_count, session_quality, ts):
    if not frame_results:
        return None

    annotated = []
    for fr in frame_results:
        img_np = cv2.cvtColor(np.array(fr["image"]), cv2.COLOR_RGB2BGR)
        count  = fr["count"]
        grade  = fr["quality"]["grade"]
        score  = fr["quality"]["overall_score"]
        is_max = fr.get("is_max", False)
        q_color = ((0,200,0) if grade == "GOOD" else
                   (0,165,255) if grade == "MARGINAL" else (0,0,220))

        # Badge
        cv2.rectangle(img_np, (0,0), (660, 95), (0,0,0), -1)
        cv2.putText(img_np,
                    f"{'★ MAX  ' if is_max else ''}Frame {fr['index']}  |  {ts.strftime('%H:%M:%S')} UTC",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,100) if is_max else (200,200,200), 1)
        cv2.putText(img_np, f"SURFERS: {count}  ({fr['confidence'].upper()})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,100) if is_max else (200,200,200), 2)
        cv2.putText(img_np,
                    f"Quality: {score:.2f} [{grade}]  |  {fr['notes'][:55]}",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.52, q_color, 1)

        scale = 640 / img_np.shape[1]
        annotated.append(cv2.resize(img_np, (640, int(img_np.shape[0] * scale))))

    max_h  = max(a.shape[0] for a in annotated)
    padded = [cv2.copyMakeBorder(a, 0, max_h - a.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(20,20,20))
              for a in annotated]
    sheet  = np.hstack(padded)

    bar = np.zeros((65, sheet.shape[1], 3), dtype=np.uint8)
    q_c = ((0,200,0) if session_quality >= 0.75 else
           (0,165,255) if session_quality >= 0.62 else (0,0,220))
    cv2.putText(bar,
                f"FINAL COUNT: {final_count}  (best-frame, 1 API call)"
                f"   |   Session quality: {session_quality:.2f}   |   {CLAUDE_MODEL}",
                (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, q_c, 2)

    out_path = OUTPUT_DIR / "burst_contact_sheet.jpg"
    cv2.imwrite(str(out_path), np.vstack([sheet, bar]))
    return out_path

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", nargs="+", help="Existing frame paths")
    parser.add_argument("--single", action="store_true", help="First frame only")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗  Set ANTHROPIC_API_KEY first.")
        print("   export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    now    = datetime.utcnow()

    print("=" * 62)
    print("SURF CROWD MONITOR — v6  (full-frame Claude vision)")
    print(f"Spot:   {TEST_SPOT}")
    print(f"Time:   {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Model:  {CLAUDE_MODEL}")
    print(f"Burst:  {BURST_FRAME_COUNT} frames × {BURST_INTERVAL_SECONDS}s")
    print("=" * 62)

    # ── 1. Frames ─────────────────────────────────────────────────────────
    if args.local:
        frame_paths = [Path(p) for p in args.local]
        if args.single:
            frame_paths = frame_paths[:1]
        print(f"\n[FETCH] Using {len(frame_paths)} local frame(s)")
    else:
        frame_paths = fetch_burst(TEST_SPOT)
        if args.single:
            frame_paths = frame_paths[:1]

    if not frame_paths:
        print("✗ No frames"); sys.exit(1)

    # ── 2. Score all frames ───────────────────────────────────────────────
    print(f"\n[PROCESS] {len(frame_paths)} frame(s)...")
    frame_results = []

    for i, fp in enumerate(frame_paths):
        label   = f"F{i+1}"
        img_pil = Image.open(fp).convert("RGB")
        quality = score_frame_quality(img_pil, label)

        print(f"\n  {label}: quality={quality['overall_score']:.2f} [{quality['grade']}]  "
              f"Lap={quality['lap_var']:.0f}  Noise={quality['noisy_pct']:.0f}%")
        if not quality["contributes"]:
            print(f"  → Below threshold ({BURST_MIN_QUALITY}) — EXCLUDED")

        frame_results.append({
            "index": i+1, "path": str(fp), "image": img_pil,
            "quality": quality, "count": -1,
            "confidence": "n/a", "notes": "excluded: low quality"
        })

    # ── 3. Claude vision on the best qualifying frame only ────────────────
    qualifying = [fr for fr in frame_results if fr["quality"]["contributes"]]
    if qualifying:
        best_qual = max(qualifying, key=lambda x: x["quality"]["overall_score"])
        label = f"F{best_qual['index']}"
        print(f"\n  Best qualifying: {label} (quality={best_qual['quality']['overall_score']:.2f})")
        print(f"  → Sending to Claude...", end=" ", flush=True)
        count, conf, notes, _ = count_surfers_claude(best_qual["image"], label, client)
        print(f"count={count}  conf={conf}")
        print(f"  → {notes}")
        best_qual.update({"count": count, "confidence": conf, "notes": notes})
    else:
        print(f"\n  All frames below threshold — no Claude call")

    # ── 4. Final count ────────────────────────────────────────────────────
    contributing = [fr for fr in frame_results
                    if fr["count"] >= 0 and fr["quality"]["contributes"]]

    if not contributing:
        final_count    = 0
        count_reliable = False
        print("\n⚠  No qualifying frames")
    else:
        best           = max(contributing, key=lambda x: x["count"])
        final_count    = best["count"]
        count_reliable = True
        best["is_max"] = True
        counts         = [fr["count"] for fr in frame_results]
        print(f"\n  Per-frame counts: {counts}")
        print(f"  Final count: {final_count}  (max — Frame {best['index']})")

    session_quality = round(
        sum(fr["quality"]["overall_score"] for fr in contributing) / len(contributing), 3
    ) if contributing else 0.0

    # ── 5. Contact sheet ──────────────────────────────────────────────────
    sheet_path = save_contact_sheet(frame_results, final_count, session_quality, now)
    if sheet_path:
        print(f"\n  Contact sheet → {sheet_path}")

    # ── 6. DB record ──────────────────────────────────────────────────────
    record = {
        "spot":            TEST_SPOT,
        "timestamp":       now.isoformat(),
        "surfer_count":    final_count,
        "count_reliable":  count_reliable,
        "count_method":    "claude_vision_full_frame_best_frame",
        "model":           CLAUDE_MODEL,
        "burst_config": {
            "frame_count":      BURST_FRAME_COUNT,
            "interval_seconds": BURST_INTERVAL_SECONDS,
            "min_quality":      BURST_MIN_QUALITY,
        },
        "session_quality": session_quality,
        "frames": [
            {
                "index":      fr["index"],
                "count":      fr["count"],
                "confidence": fr["confidence"],
                "notes":      fr["notes"],
                "is_max":     fr.get("is_max", False),
                "quality": {
                    "score":     fr["quality"]["overall_score"],
                    "grade":     fr["quality"]["grade"],
                    "lap_var":   fr["quality"]["lap_var"],
                    "noisy_pct": fr["quality"]["noisy_pct"],
                }
            }
            for fr in frame_results
        ]
    }

    print("\n" + "=" * 62)
    print("DB RECORD")
    print("=" * 62)
    print(json.dumps(record, indent=2))
    print("=" * 62)
