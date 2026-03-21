"""
Surf Crowd Monitor — Spike v3
==============================
Adds frame quality scoring before detection runs.
Quality score (0.0–1.0) reflects how usable a frame is for CV detection.
Low quality = glare, darkness, fog, or blown-out highlights.
High quality = flat light, good contrast, surfers visible.

A surfer_count from a low-quality frame is flagged as unreliable.
A surfer_count from a high-quality frame is considered trustworthy.

Quality dimensions scored:
  - Glare index       (specular highlight density in water zone)
  - Brightness        (overall exposure — too dark or too bright = bad)
  - Contrast          (low contrast = fog, haze, or flat/dark conditions)
  - Saturation        (desaturated = overcast or underexposed)

Requirements:
  pip3 install requests opencv-python ultralytics pillow

Usage:
  python3 surf_spike_v3.py                          # fetch fresh frame + detect
  python3 surf_spike_v3.py --local spike_output/frame.jpg   # reuse existing frame
  python3 surf_spike_v3.py --local spike_output/frame.jpg --quality-only  # skip detection
"""

import requests
import subprocess
import os
import sys
import json
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

# Water zone as fraction of frame height (top, bottom)
# Crops out sky and beach before quality scoring and detection
WATER_ZONE = {
    "lower_trestles": (0.28, 0.65),
    "hb_pier_south":  (0.25, 0.65),
    "hb_cliffs":      (0.25, 0.65),
    "newport_56th":   (0.25, 0.65),
}

# Quality score thresholds
QUALITY_UNUSABLE  = 0.35   # below this: don't trust count at all
QUALITY_MARGINAL  = 0.60   # below this: count logged but flagged
QUALITY_GOOD      = 0.75   # above this: count is reliable

# Glare detection — pixel brightness threshold for "blown out"
GLARE_PIXEL_THRESH   = 220   # 0–255, pixels above this count as glare
GLARE_BAD_FRACTION   = 0.25  # if >25% of water zone is glare = bad frame

# Detection config
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

def fetch_frame(spot_key):
    url = SPOTS[spot_key]
    print(f"\n[1] Fetching playlist: {spot_key}")
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"    ✗ {r.status_code}")
        return None
    lines = [l.strip() for l in r.text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    seg_url = lines[-1] if lines[-1].startswith("http") else f"{url.rsplit('/',1)[0]}/{lines[-1]}"
    print(f"    Segment: {lines[-1]}")
    r2 = requests.get(seg_url, headers=HEADERS, timeout=30)
    if r2.status_code != 200:
        return None
    seg_path = OUTPUT_DIR / "segment.ts"
    seg_path.write_bytes(r2.content)
    frame_path = OUTPUT_DIR / "frame.jpg"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(seg_path), "-frames:v", "1", "-q:v", "2", str(frame_path)],
        capture_output=True
    )
    if result.returncode != 0:
        print("    ✗ ffmpeg failed")
        return None
    print(f"    ✓ Frame saved → {frame_path}")
    return frame_path

# ─── FRAME QUALITY SCORER ─────────────────────────────────────────────────────

def score_frame_quality(img_pil, spot_key, save_debug=True):
    """
    Scores a frame on 4 dimensions, returns an overall quality score 0.0–1.0.

    Returns dict with:
      overall_score   float 0–1
      grade           str   GOOD / MARGINAL / POOR / UNUSABLE
      reliable        bool  whether detection count should be trusted
      dimensions      dict  per-dimension scores and raw values
      notes           list  human-readable explanation
    """
    print(f"\n[QC] Scoring frame quality...")

    w, h = img_pil.size
    zone_top, zone_bot = WATER_ZONE.get(spot_key, (0.25, 0.70))
    wy1 = int(h * zone_top)
    wy2 = int(h * zone_bot)

    # Work in numpy for pixel-level analysis
    water_np  = np.array(img_pil.crop((0, wy1, w, wy2)))  # RGB
    water_gray = cv2.cvtColor(water_np, cv2.COLOR_RGB2GRAY)
    water_hsv  = cv2.cvtColor(water_np, cv2.COLOR_RGB2HSV)

    notes = []
    dims  = {}

    # ── 1. GLARE INDEX ──────────────────────────────────────────────────────
    # Fraction of water zone pixels that are blown-out highlights
    glare_mask     = water_gray > GLARE_PIXEL_THRESH
    glare_fraction = float(glare_mask.sum()) / glare_mask.size

    # Score: 0 glare = 1.0, at GLARE_BAD_FRACTION or above = 0.0
    glare_score = max(0.0, 1.0 - (glare_fraction / GLARE_BAD_FRACTION))
    dims["glare"] = {
        "score":          round(glare_score, 3),
        "glare_fraction": round(glare_fraction, 3),
        "threshold":      GLARE_BAD_FRACTION,
    }
    if glare_fraction > GLARE_BAD_FRACTION:
        notes.append(f"Heavy glare — {glare_fraction*100:.1f}% of water zone blown out (threshold {GLARE_BAD_FRACTION*100:.0f}%)")
    elif glare_fraction > 0.10:
        notes.append(f"Moderate glare — {glare_fraction*100:.1f}% of water zone affected")

    # ── 2. BRIGHTNESS ───────────────────────────────────────────────────────
    # Good range: mean brightness 60–200. Too dark (<60) or too bright (>200) = bad.
    mean_brightness = float(water_gray.mean())
    if mean_brightness < 60:
        brightness_score = mean_brightness / 60.0
        notes.append(f"Frame too dark — mean brightness {mean_brightness:.0f} (minimum 60)")
    elif mean_brightness > 200:
        brightness_score = max(0.0, 1.0 - (mean_brightness - 200) / 55.0)
        notes.append(f"Frame overexposed — mean brightness {mean_brightness:.0f} (maximum 200)")
    else:
        # Peak score at ~130 (ideal), falls off toward edges of range
        brightness_score = 1.0 - abs(mean_brightness - 130) / 70.0
    brightness_score = max(0.0, min(1.0, brightness_score))
    dims["brightness"] = {
        "score":           round(brightness_score, 3),
        "mean_brightness": round(mean_brightness, 1),
        "ideal_range":     "60–200",
    }

    # ── 3. CONTRAST ─────────────────────────────────────────────────────────
    # Standard deviation of pixel brightness. Low = fog/haze/darkness.
    # Good: std > 30. Poor: std < 15.
    brightness_std = float(water_gray.std())
    contrast_score = min(1.0, brightness_std / 40.0)
    dims["contrast"] = {
        "score":          round(contrast_score, 3),
        "brightness_std": round(brightness_std, 1),
        "minimum_good":   30,
    }
    if brightness_std < 15:
        notes.append(f"Very low contrast — may be foggy, dark, or heavily overcast (std={brightness_std:.1f})")
    elif brightness_std < 30:
        notes.append(f"Low contrast — detection may be impaired (std={brightness_std:.1f})")

    # ── 4. SATURATION ───────────────────────────────────────────────────────
    # HSV saturation channel. Very desaturated = early dark, dense fog, or blown out.
    # We use a soft floor — some ocean frames legitimately have low saturation on overcast days.
    mean_sat = float(water_hsv[:,:,1].mean())  # 0–255
    sat_score = min(1.0, mean_sat / 60.0)      # full score at 60+ saturation
    dims["saturation"] = {
        "score":        round(sat_score, 3),
        "mean_sat":     round(mean_sat, 1),
        "minimum_good": 40,
    }
    if mean_sat < 20:
        notes.append(f"Very low saturation — frame may be near-dark or extremely washed out (sat={mean_sat:.1f})")

    # ── OVERALL SCORE ───────────────────────────────────────────────────────
    # Glare is weighted most heavily since it's the dominant failure mode for this use case
    weights = {
        "glare":      0.45,
        "brightness": 0.25,
        "contrast":   0.20,
        "saturation": 0.10,
    }
    overall = sum(dims[k]["score"] * weights[k] for k in weights)
    overall = round(overall, 3)

    # Grade
    if overall >= QUALITY_GOOD:
        grade    = "GOOD"
        reliable = True
        emoji    = "✓"
    elif overall >= QUALITY_MARGINAL:
        grade    = "MARGINAL"
        reliable = True
        emoji    = "~"
        notes.append("Detection count logged but marked as marginal quality")
    elif overall >= QUALITY_UNUSABLE:
        grade    = "POOR"
        reliable = False
        emoji    = "✗"
        notes.append("Detection count unreliable — frame quality too low")
    else:
        grade    = "UNUSABLE"
        reliable = False
        emoji    = "✗✗"
        notes.append("Frame unusable — count will NOT be stored")

    if not notes:
        notes.append("Frame looks clean — good conditions for detection")

    print(f"    Glare score:      {dims['glare']['score']:.2f}  (glare={dims['glare']['glare_fraction']*100:.1f}%)")
    print(f"    Brightness score: {dims['brightness']['score']:.2f}  (mean={dims['brightness']['mean_brightness']:.0f})")
    print(f"    Contrast score:   {dims['contrast']['score']:.2f}  (std={dims['contrast']['brightness_std']:.1f})")
    print(f"    Saturation score: {dims['saturation']['score']:.2f}  (sat={dims['saturation']['mean_sat']:.1f})")
    print(f"    ─────────────────────────────────")
    print(f"    Overall quality:  {overall:.2f}  [{grade}]  {emoji}")
    for note in notes:
        print(f"    → {note}")

    # ── DEBUG VISUALIZATION ─────────────────────────────────────────────────
    if save_debug:
        debug = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # Glare overlay — highlight blown-out pixels in red
        water_bgr = cv2.cvtColor(water_np, cv2.COLOR_RGB2BGR)
        glare_overlay = water_bgr.copy()
        glare_overlay[glare_mask] = (0, 0, 220)
        blended = cv2.addWeighted(water_bgr, 0.65, glare_overlay, 0.35, 0)
        debug[wy1:wy2, 0:w] = blended
        # Quality badge
        badge_color = (0,200,0) if grade=="GOOD" else (0,165,255) if grade=="MARGINAL" else (0,0,220)
        cv2.rectangle(debug, (0,0), (420, 58), (0,0,0), -1)
        cv2.putText(debug, f"QUALITY: {overall:.2f}  [{grade}]",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, badge_color, 2)
        cv2.rectangle(debug, (0, wy1), (w, wy2), (0,165,255), 2)
        cv2.putText(debug, "SCORED ZONE", (12, wy1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)
        debug_path = OUTPUT_DIR / "frame_quality_debug.jpg"
        cv2.imwrite(str(debug_path), debug)
        print(f"    Debug frame → {debug_path}  (red = glare pixels)")

    return {
        "overall_score": overall,
        "grade":         grade,
        "reliable":      reliable,
        "weights":       weights,
        "dimensions":    dims,
        "notes":         notes,
    }

# ─── TILED DETECTION (unchanged from v2) ─────────────────────────────────────

def iou(a, b):
    ax1,ay1,ax2,ay2,_ = a; bx1,by1,bx2,by2,_ = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter=(ix2-ix1)*(iy2-iy1)
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6)

def run_tiled_detection(frame_path, spot_key):
    print(f"\n[5] Running tiled detection...")
    img  = Image.open(frame_path).convert("RGB")
    w, h = img.size
    zone_top, zone_bot = WATER_ZONE.get(spot_key, (0.25, 0.70))
    wy1 = int(h * zone_top); wy2 = int(h * zone_bot)
    water = img.crop((0, wy1, w, wy2))
    ww, wh = water.size
    tile_w = int(ww / (TILE_COLS - TILE_OVERLAP * (TILE_COLS-1)))
    tile_h = int(wh / (TILE_ROWS - TILE_OVERLAP * (TILE_ROWS-1)))
    step_x = int(tile_w * (1 - TILE_OVERLAP))
    step_y = int(tile_h * (1 - TILE_OVERLAP))
    model  = YOLO("yolov8n.pt")
    all_dets = []
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            tx1=col*step_x; ty1=row*step_y
            tx2=min(tx1+tile_w,ww); ty2=min(ty1+tile_h,wh)
            tile = ImageEnhance.Contrast(water.crop((tx1,ty1,tx2,ty2))).enhance(CONTRAST_BOOST)
            results = model(np.array(tile), classes=[0], conf=CONF_THRESHOLD, verbose=False)
            for box in results[0].boxes:
                bx1,by1,bx2,by2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                all_dets.append((bx1+tx1, by1+ty1+wy1, bx2+tx1, by2+ty1+wy1, conf))
    sorted_dets = sorted(all_dets, key=lambda x: x[4], reverse=True)
    keep = []
    for det in sorted_dets:
        if all(iou(det,k) < NMS_IOU_THRESH for k in keep):
            keep.append(det)
    print(f"    Raw: {len(all_dets)}  After NMS: {len(keep)}")
    return keep, img, (wy1, wy2)

# ─── ANNOTATE ─────────────────────────────────────────────────────────────────

def save_annotated(img, detections, water_zone_y, quality, spot_key, timestamp):
    wy1, wy2 = water_zone_y
    w, h = img.size
    full_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(full_np, (0, wy1), (w, wy2), (0,165,255), 2)
    for (x1,y1,x2,y2,conf) in detections:
        cv2.rectangle(full_np, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(full_np, f"{conf:.2f}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
    # Count + quality badge
    grade = quality["grade"]
    overall = quality["overall_score"]
    q_color = (0,200,0) if grade=="GOOD" else (0,165,255) if grade=="MARGINAL" else (0,0,220)
    cv2.rectangle(full_np, (0,0), (500, 90), (0,0,0), -1)
    cv2.putText(full_np, f"SURFERS: {len(detections)}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)
    cv2.putText(full_np, f"FRAME QUALITY: {overall:.2f}  [{grade}]",
                (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, q_color, 2)
    cv2.putText(full_np, timestamp.strftime("%Y-%m-%d %H:%M UTC"),
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    out_path = OUTPUT_DIR / "frame_final.jpg"
    cv2.imwrite(str(out_path), full_np)
    print(f"\n    Annotated frame → {out_path}")
    return out_path

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="Use existing frame instead of fetching")
    parser.add_argument("--quality-only", action="store_true", help="Score quality only, skip detection")
    args = parser.parse_args()

    now = datetime.utcnow()
    print("=" * 58)
    print("SURF CROWD MONITOR — SPIKE v3  (with frame quality scoring)")
    print(f"Spot:  {TEST_SPOT}")
    print(f"Time:  {now.isoformat()}Z")
    print("=" * 58)

    # 1. Get frame
    if args.local:
        frame_path = Path(args.local)
        print(f"\nUsing local frame: {frame_path}")
    else:
        frame_path = fetch_frame(TEST_SPOT)
        if not frame_path: sys.exit(1)

    img_pil = Image.open(frame_path).convert("RGB")

    # 2. Score quality first
    quality = score_frame_quality(img_pil, TEST_SPOT)

    if args.quality_only:
        print("\n[--quality-only mode, skipping detection]")
        sys.exit(0)

    # 3. Detect (even on poor frames for spike purposes — in prod we'd skip)
    if not quality["reliable"]:
        print(f"\n⚠  Frame quality is {quality['grade']} — detection count will be marked unreliable")
        print("   (Running anyway for spike validation purposes)")

    detections, img, water_zone_y = run_tiled_detection(frame_path, TEST_SPOT)
    out_path = save_annotated(img, detections, water_zone_y, quality, TEST_SPOT, now)

    # 4. Final record — what would actually be written to the DB
    record = {
        "spot":            TEST_SPOT,
        "timestamp":       now.isoformat(),
        "surfer_count":    len(detections),
        "count_reliable":  quality["reliable"],
        "frame_quality": {
            "overall_score": quality["overall_score"],
            "grade":         quality["grade"],
            "glare_pct":     round(quality["dimensions"]["glare"]["glare_fraction"] * 100, 1),
            "brightness":    quality["dimensions"]["brightness"]["mean_brightness"],
            "contrast_std":  quality["dimensions"]["contrast"]["brightness_std"],
        },
        "notes": quality["notes"],
    }

    print("\n" + "=" * 58)
    print("DB RECORD (what would be written to Postgres)")
    print("=" * 58)
    print(json.dumps(record, indent=2))
    print(f"\n→ {out_path}  — final annotated frame")
    print(f"→ spike_output/frame_quality_debug.jpg  — glare heatmap")
    print("=" * 58)
