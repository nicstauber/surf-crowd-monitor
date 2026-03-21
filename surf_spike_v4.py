"""
Surf Crowd Monitor — Spike v4
==============================
Adds multi-frame burst sampling per interval.

Instead of a single frame, each sampling cycle captures a configurable
burst of frames spaced BURST_INTERVAL_SECONDS apart. Detection runs on
each frame independently. The final reported count is the MAX across all
frames in the burst — catching surfers that were hidden behind waves in
any individual frame.

Each frame in the burst gets its own quality score. The final record
includes per-frame detail plus a rollup. Only frames that meet minimum
quality thresholds contribute to the max count.

Burst config (all configurable below):
  BURST_FRAME_COUNT       = 3       frames per interval
  BURST_INTERVAL_SECONDS  = 10      seconds between frames
  BURST_MIN_QUALITY       = 0.45    frames below this grade are excluded from count

Usage:
  python3 surf_spike_v4.py                               # fetch fresh burst
  python3 surf_spike_v4.py --local f1.jpg f2.jpg f3.jpg  # test on local frames
  python3 surf_spike_v4.py --quality-only                # quality scoring only
"""

import requests
import subprocess
import sys
import json
import time
import argparse
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SPOTS = {
    "lower_trestles": "https://hls.cdn-surfline.com/oregon/wc-lowerslefts/playlist.m3u8",
    "hb_pier_south":  "https://hls.cdn-surfline.com/oregon/wc-huntingtonbeachsouthside/playlist.m3u8",
    "hb_cliffs":      "https://hls.cdn-surfline.com/oregon/wc-huntingtoncliffs/playlist.m3u8",
    "newport_56th":   "https://hls.cdn-surfline.com/oregon/wc-56thstreet/playlist.m3u8",
}
TEST_SPOT = "lower_trestles"

# ── Burst sampling ────────────────────────────────────────────────────────────
BURST_FRAME_COUNT      = 3     # frames captured per sampling interval
BURST_INTERVAL_SECONDS = 10    # seconds to wait between each frame capture
BURST_MIN_QUALITY      = 0.45  # frames below this score don't contribute to count

# ── Water zone (fraction of frame height) ────────────────────────────────────
WATER_ZONE = {
    "lower_trestles": (0.28, 0.65),
    "hb_pier_south":  (0.25, 0.65),
    "hb_cliffs":      (0.25, 0.65),
    "newport_56th":   (0.25, 0.65),
}

# ── Quality thresholds (calibrated from real frame data) ─────────────────────
LAP_VAR_CLEAN       = 800
LAP_VAR_BAD         = 4000
PATCH_NOISE_CLEAN   = 0.25
PATCH_NOISE_BAD     = 0.70
BRIGHTNESS_MIN      = 60
BRIGHTNESS_MAX      = 200
BRIGHTNESS_IDEAL    = 115
QUALITY_UNUSABLE    = 0.30
QUALITY_POOR        = 0.45
QUALITY_MARGINAL    = 0.62
QUALITY_GOOD        = 0.75

# ── Detection ─────────────────────────────────────────────────────────────────
TILE_COLS      = 6
TILE_ROWS      = 2
TILE_OVERLAP   = 0.20
CONF_THRESHOLD = 0.20
NMS_IOU_THRESH = 0.30
CONTRAST_BOOST = 1.5

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
    """Fetch one frame from the HLS stream. Returns Path or None."""
    url = SPOTS[spot_key]
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"    ✗ Playlist {r.status_code}")
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
    if res.returncode != 0 or not frame_path.exists():
        return None

    return frame_path


def fetch_burst(spot_key):
    """
    Capture BURST_FRAME_COUNT frames spaced BURST_INTERVAL_SECONDS apart.
    Returns list of Paths (may be shorter than BURST_FRAME_COUNT on errors).
    """
    print(f"\n[FETCH] Burst capture: {BURST_FRAME_COUNT} frames × {BURST_INTERVAL_SECONDS}s apart")
    frames = []
    for i in range(BURST_FRAME_COUNT):
        if i > 0:
            print(f"  Waiting {BURST_INTERVAL_SECONDS}s before frame {i+1}...")
            time.sleep(BURST_INTERVAL_SECONDS)
        print(f"  Capturing frame {i+1}/{BURST_FRAME_COUNT}...", end=" ")
        path = fetch_single_frame(spot_key, i + 1)
        if path:
            print(f"✓ → {path.name}")
            frames.append(path)
        else:
            print(f"✗ failed")
    print(f"  Burst complete: {len(frames)}/{BURST_FRAME_COUNT} frames captured")
    return frames

# ─── FRAME QUALITY SCORER ─────────────────────────────────────────────────────

def score_frame_quality(img_pil, spot_key, frame_label="", save_debug=False):
    """Score a single frame. Returns quality dict."""
    w, h = img_pil.size
    zone_top, zone_bot = WATER_ZONE.get(spot_key, (0.25, 0.70))
    wy1 = int(h * zone_top)
    wy2 = int(h * zone_bot)

    water_np   = np.array(img_pil.crop((0, wy1, w, wy2)))
    water_gray = cv2.cvtColor(water_np, cv2.COLOR_RGB2GRAY)

    # Texture / glare
    lap      = cv2.Laplacian(water_gray, cv2.CV_64F)
    lap_var  = float(lap.var())
    lap_score = 1.0 - min(1.0, max(0.0, (lap_var - LAP_VAR_CLEAN) / (LAP_VAR_BAD - LAP_VAR_CLEAN)))

    patch_stds = [
        float(water_gray[y:y+16, x:x+16].std())
        for y in range(0, water_gray.shape[0]-16, 16)
        for x in range(0, water_gray.shape[1]-16, 16)
    ]
    noisy_frac  = float(np.mean(np.array(patch_stds) > 20))
    patch_score = 1.0 - min(1.0, max(0.0, (noisy_frac - PATCH_NOISE_CLEAN) / (PATCH_NOISE_BAD - PATCH_NOISE_CLEAN)))
    glare_score = (lap_score + patch_score) / 2.0

    # Brightness
    mean_brightness = float(water_gray.mean())
    if mean_brightness < BRIGHTNESS_MIN:
        brightness_score = mean_brightness / BRIGHTNESS_MIN
    elif mean_brightness > BRIGHTNESS_MAX:
        brightness_score = max(0.0, 1.0 - (mean_brightness - BRIGHTNESS_MAX) / 55.0)
    else:
        brightness_score = 1.0 - abs(mean_brightness - BRIGHTNESS_IDEAL) / 80.0
    brightness_score = max(0.0, min(1.0, brightness_score))

    # Contrast
    std = float(water_gray.std())
    contrast_score = min(1.0, std / 25.0) if std < 25 else 1.0

    # Overall
    overall = round(
        glare_score     * 0.55 +
        brightness_score * 0.25 +
        contrast_score  * 0.20,
        3
    )

    if overall >= QUALITY_GOOD:
        grade = "GOOD";     reliable = True
    elif overall >= QUALITY_MARGINAL:
        grade = "MARGINAL"; reliable = True
    elif overall >= QUALITY_POOR:
        grade = "POOR";     reliable = False
    else:
        grade = "UNUSABLE"; reliable = False

    # Save debug heatmap if requested
    if save_debug:
        debug    = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        lap_abs  = np.abs(lap).astype(np.float32)
        lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lap_col  = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)
        blended  = cv2.addWeighted(cv2.cvtColor(water_np, cv2.COLOR_RGB2BGR), 0.55, lap_col, 0.45, 0)
        debug[wy1:wy2, 0:w] = blended
        q_color  = (0,200,0) if grade=="GOOD" else (0,165,255) if grade=="MARGINAL" else (0,0,220)
        cv2.rectangle(debug, (0,0), (700, 62), (0,0,0), -1)
        cv2.putText(debug, f"[{frame_label}] QUALITY: {overall:.2f} [{grade}]  Lap={lap_var:.0f}  Noise={noisy_frac*100:.0f}%",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, q_color, 2)
        debug_path = OUTPUT_DIR / f"debug_quality_{frame_label}.jpg"
        cv2.imwrite(str(debug_path), debug)

    return {
        "overall_score":    overall,
        "grade":            grade,
        "reliable":         reliable,
        "contributes":      overall >= BURST_MIN_QUALITY,
        "lap_var":          round(lap_var, 1),
        "noisy_pct":        round(noisy_frac * 100, 1),
        "mean_brightness":  round(mean_brightness, 1),
        "water_zone":       (wy1, wy2),
    }

# ─── DETECTION ───────────────────────────────────────────────────────────────

def iou(a, b):
    ax1,ay1,ax2,ay2,_ = a; bx1,by1,bx2,by2,_ = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter=(ix2-ix1)*(iy2-iy1)
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6)


def detect_frame(img_pil, spot_key, model):
    """Run tiled detection on a single PIL image. Returns list of detections."""
    w, h  = img_pil.size
    zone_top, zone_bot = WATER_ZONE.get(spot_key, (0.25, 0.70))
    wy1 = int(h * zone_top)
    wy2 = int(h * zone_bot)
    water = img_pil.crop((0, wy1, w, wy2))
    ww, wh = water.size

    tile_w = int(ww / (TILE_COLS - TILE_OVERLAP * (TILE_COLS - 1)))
    tile_h = int(wh / (TILE_ROWS - TILE_OVERLAP * (TILE_ROWS - 1)))
    step_x = int(tile_w * (1 - TILE_OVERLAP))
    step_y = int(tile_h * (1 - TILE_OVERLAP))

    all_dets = []
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            tx1 = col * step_x; ty1 = row * step_y
            tx2 = min(tx1 + tile_w, ww); ty2 = min(ty1 + tile_h, wh)
            tile = ImageEnhance.Contrast(water.crop((tx1, ty1, tx2, ty2))).enhance(CONTRAST_BOOST)
            results = model(np.array(tile), classes=[0], conf=CONF_THRESHOLD, verbose=False)
            for box in results[0].boxes:
                bx1,by1,bx2,by2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                all_dets.append((bx1+tx1, by1+ty1+wy1, bx2+tx1, by2+ty1+wy1, conf))

    keep = []
    for det in sorted(all_dets, key=lambda x: x[4], reverse=True):
        if all(iou(det, k) < NMS_IOU_THRESH for k in keep):
            keep.append(det)
    return keep

# ─── ANNOTATED BURST CONTACT SHEET ───────────────────────────────────────────

def save_burst_contact_sheet(frame_results, final_count, overall_quality, ts):
    """
    Saves a side-by-side contact sheet of all burst frames with their
    individual counts and quality scores. Makes it easy to visually verify.
    """
    n = len(frame_results)
    if n == 0:
        return None

    # Load and annotate each frame
    annotated = []
    for fr in frame_results:
        img_np = cv2.cvtColor(np.array(fr["image"]), cv2.COLOR_RGB2BGR)
        wy1, wy2 = fr["quality"]["water_zone"]
        w = img_np.shape[1]

        # Detection boxes
        for (x1,y1,x2,y2,conf) in fr["detections"]:
            cv2.rectangle(img_np, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

        # Water zone boundary
        cv2.rectangle(img_np, (0,wy1), (w,wy2), (0,165,255), 1)

        # Per-frame badge
        grade   = fr["quality"]["grade"]
        score   = fr["quality"]["overall_score"]
        count   = len(fr["detections"])
        contrib = fr["quality"]["contributes"]
        q_color = (0,200,0) if grade=="GOOD" else (0,165,255) if grade=="MARGINAL" else (0,0,220)
        flag    = "" if contrib else " [EXCLUDED]"
        is_max  = "★ MAX  " if fr.get("is_max") else ""

        cv2.rectangle(img_np, (0,0), (620, 95), (0,0,0), -1)
        cv2.putText(img_np, f"Frame {fr['index']}  |  {ts.strftime('%H:%M:%S')} UTC",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200,200,200), 1)
        cv2.putText(img_np, f"{is_max}SURFERS: {count}{flag}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0,255,100) if fr.get("is_max") else (200,200,200), 2)
        cv2.putText(img_np, f"Quality: {score:.2f} [{grade}]  Lap={fr['quality']['lap_var']:.0f}  Noise={fr['quality']['noisy_pct']:.0f}%",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62, q_color, 1)

        # Scale down for contact sheet (each frame → 640px wide)
        scale = 640 / img_np.shape[1]
        small = cv2.resize(img_np, (640, int(img_np.shape[0] * scale)))
        annotated.append(small)

    # Stack horizontally
    # Pad to same height first
    max_h = max(a.shape[0] for a in annotated)
    padded = []
    for a in annotated:
        pad = max_h - a.shape[0]
        padded.append(cv2.copyMakeBorder(a, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20)))

    sheet = np.hstack(padded)

    # Summary bar at bottom
    bar_h = 60
    bar = np.zeros((bar_h, sheet.shape[1], 3), dtype=np.uint8)
    grade_color = (0,200,0) if overall_quality >= QUALITY_GOOD else \
                  (0,165,255) if overall_quality >= QUALITY_MARGINAL else (0,0,220)
    cv2.putText(bar, f"FINAL COUNT: {final_count}  (max across {n} frames)   |   Session quality: {overall_quality:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, grade_color, 2)

    final = np.vstack([sheet, bar])
    out_path = OUTPUT_DIR / "burst_contact_sheet.jpg"
    cv2.imwrite(str(out_path), final)
    return out_path

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", nargs="+", help="Paths to existing frames (e.g. --local f1.jpg f2.jpg f3.jpg)")
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()

    now = datetime.utcnow()
    print("=" * 62)
    print("SURF CROWD MONITOR — v4  (multi-frame burst sampling)")
    print(f"Spot:  {TEST_SPOT}")
    print(f"Time:  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Burst: {BURST_FRAME_COUNT} frames × {BURST_INTERVAL_SECONDS}s  |  min quality: {BURST_MIN_QUALITY}")
    print("=" * 62)

    # ── 1. Get frames ──────────────────────────────────────────────────────
    if args.local:
        frame_paths = [Path(p) for p in args.local]
        print(f"\n[FETCH] Using {len(frame_paths)} local frames")
    else:
        frame_paths = fetch_burst(TEST_SPOT)

    if not frame_paths:
        print("✗ No frames captured"); sys.exit(1)

    # Load model once, reuse across all frames
    print(f"\n[MODEL] Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print(f"        ✓ Ready")

    # ── 2. Score + detect each frame ───────────────────────────────────────
    print(f"\n[BURST] Processing {len(frame_paths)} frames...")
    frame_results = []

    for i, fp in enumerate(frame_paths):
        label = f"F{i+1}"
        img_pil = Image.open(fp).convert("RGB")

        quality = score_frame_quality(img_pil, TEST_SPOT,
                                      frame_label=label, save_debug=True)

        if args.quality_only:
            print(f"  {label}: quality={quality['overall_score']:.2f} [{quality['grade']}]  "
                  f"Lap={quality['lap_var']:.0f}  Noise={quality['noisy_pct']:.0f}%  "
                  f"{'✓ contributes' if quality['contributes'] else '✗ excluded'}")
            continue

        if not quality["contributes"]:
            print(f"  {label}: quality={quality['overall_score']:.2f} [{quality['grade']}] — "
                  f"below threshold {BURST_MIN_QUALITY}, EXCLUDED from count")
            detections = []
        else:
            detections = detect_frame(img_pil, TEST_SPOT, model)
            print(f"  {label}: quality={quality['overall_score']:.2f} [{quality['grade']}]  "
                  f"surfers={len(detections)}  "
                  f"Lap={quality['lap_var']:.0f}  Noise={quality['noisy_pct']:.0f}%")

        frame_results.append({
            "index":      i + 1,
            "path":       str(fp),
            "image":      img_pil,
            "quality":    quality,
            "detections": detections,
            "count":      len(detections),
        })

    if args.quality_only:
        sys.exit(0)

    # ── 3. Determine final count ───────────────────────────────────────────
    contributing = [fr for fr in frame_results if fr["quality"]["contributes"]]

    if not contributing:
        final_count   = 0
        count_reliable = False
        max_frame_idx  = None
        print(f"\n⚠  No frames met quality threshold {BURST_MIN_QUALITY} — count unreliable")
    else:
        best_frame    = max(contributing, key=lambda x: x["count"])
        final_count   = best_frame["count"]
        count_reliable = True
        max_frame_idx  = best_frame["index"]
        best_frame["is_max"] = True
        print(f"\n  Final count: {final_count}  (max from Frame {max_frame_idx})")
        print(f"  Counts per frame: {[fr['count'] for fr in frame_results]}")

    # ── 4. Overall session quality (mean of contributing frame scores) ─────
    overall_quality = round(
        sum(fr["quality"]["overall_score"] for fr in contributing) / len(contributing), 3
    ) if contributing else 0.0

    # ── 5. Save contact sheet ─────────────────────────────────────────────
    sheet_path = save_burst_contact_sheet(frame_results, final_count, overall_quality, now)
    if sheet_path:
        print(f"\n  Contact sheet → {sheet_path}")

    # ── 6. DB record ──────────────────────────────────────────────────────
    record = {
        "spot":             TEST_SPOT,
        "timestamp":        now.isoformat(),
        "surfer_count":     final_count,
        "count_reliable":   count_reliable,
        "count_method":     "max_across_burst",
        "burst_config": {
            "frame_count":       BURST_FRAME_COUNT,
            "interval_seconds":  BURST_INTERVAL_SECONDS,
            "min_quality":       BURST_MIN_QUALITY,
        },
        "session_quality":  overall_quality,
        "frames": [
            {
                "index":       fr["index"],
                "count":       fr["count"],
                "contributed": fr["quality"]["contributes"],
                "is_max":      fr.get("is_max", False),
                "quality": {
                    "score":      fr["quality"]["overall_score"],
                    "grade":      fr["quality"]["grade"],
                    "lap_var":    fr["quality"]["lap_var"],
                    "noisy_pct":  fr["quality"]["noisy_pct"],
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
