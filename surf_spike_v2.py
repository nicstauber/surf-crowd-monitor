"""
Surf Crowd Monitor — Tiled Detection Spike v2
==============================================
Improvements over v1:
  - Slices the water zone into overlapping tiles before detection
  - Each surfer appears ~3x larger relative to tile = much better detection
  - NMS deduplication removes double-counts from overlapping tiles
  - Contrast boost per tile to fight afternoon glare
  - Water zone is configurable per cam angle

Requirements:
  pip3 install requests opencv-python ultralytics pillow

Usage:
  python3 surf_spike_v2.py

  Or to re-run detection on an existing frame (skip fetch):
  python3 surf_spike_v2.py --local spike_output/frame.jpg
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
    "lower_trestles":  "https://hls.cdn-surfline.com/oregon/wc-lowerslefts/playlist.m3u8",
    "hb_pier_south":   "https://hls.cdn-surfline.com/oregon/wc-huntingtonbeachsouthside/playlist.m3u8",
    "hb_cliffs":       "https://hls.cdn-surfline.com/oregon/wc-huntingtoncliffs/playlist.m3u8",
    "newport_56th":    "https://hls.cdn-surfline.com/oregon/wc-56thstreet/playlist.m3u8",
}

TEST_SPOT = "lower_trestles"

# Water zone — crops out sky and beach before tiling
# Tune these per cam: (top_px, bottom_px) as fraction of frame height
# 0.0 = top of frame, 1.0 = bottom of frame
WATER_ZONE = {
    "lower_trestles": (0.28, 0.65),  # tuned from spike frame
    "hb_pier_south":  (0.25, 0.65),  # placeholder — tune after first capture
    "hb_cliffs":      (0.25, 0.65),
    "newport_56th":   (0.25, 0.65),
}

# Tiling config
TILE_COLS       = 6      # horizontal tiles across water zone
TILE_ROWS       = 2      # vertical rows
TILE_OVERLAP    = 0.20   # 20% overlap between tiles to avoid edge misses
CONF_THRESHOLD  = 0.20   # lower than default (0.25) to catch small surfers
NMS_IOU_THRESH  = 0.30   # boxes overlapping > 30% are considered duplicates
CONTRAST_BOOST  = 1.5    # applied per tile before detection

HEADERS = {
    "origin":     "https://www.surfline.com",
    "referer":    "https://www.surfline.com/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "accept":     "*/*",
}

OUTPUT_DIR = Path("./spike_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── STREAM FETCH (same as v1) ────────────────────────────────────────────────

def fetch_frame(spot_key):
    url = SPOTS[spot_key]
    print(f"\n[1] Fetching playlist for {spot_key}...")
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"    ✗ Playlist fetch failed: {r.status_code}")
        return None

    lines = [l.strip() for l in r.text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        print("    ✗ No segments in playlist")
        return None

    latest = lines[-1]
    base   = url.rsplit("/", 1)[0]
    seg_url = latest if latest.startswith("http") else f"{base}/{latest}"
    print(f"    Segment: {latest}")

    print(f"[2] Downloading segment...")
    r2 = requests.get(seg_url, headers=HEADERS, timeout=30)
    if r2.status_code != 200:
        print(f"    ✗ Segment fetch failed")
        return None
    seg_path = OUTPUT_DIR / "segment.ts"
    seg_path.write_bytes(r2.content)
    print(f"    ✓ {len(r2.content)/1024:.1f} KB")

    print(f"[3] Extracting frame...")
    frame_path = OUTPUT_DIR / "frame.jpg"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(seg_path), "-frames:v", "1", "-q:v", "2", str(frame_path)],
        capture_output=True
    )
    if result.returncode != 0 or not frame_path.exists():
        print(f"    ✗ ffmpeg failed")
        return None
    print(f"    ✓ Frame saved → {frame_path}")
    return frame_path

# ─── TILED DETECTION ─────────────────────────────────────────────────────────

def iou(a, b):
    """Intersection over Union for two boxes (x1,y1,x2,y2,conf)."""
    ax1,ay1,ax2,ay2,_ = a
    bx1,by1,bx2,by2,_ = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2-ix1) * (iy2-iy1)
    a_area = (ax2-ax1) * (ay2-ay1)
    b_area = (bx2-bx1) * (by2-by1)
    return inter / (a_area + b_area - inter + 1e-6)

def run_tiled_detection(frame_path, spot_key):
    print(f"\n[4] Running tiled detection...")

    img  = Image.open(frame_path).convert("RGB")
    w, h = img.size
    print(f"    Frame: {w}x{h}")

    # Water zone in pixels
    zone_top, zone_bot = WATER_ZONE.get(spot_key, (0.25, 0.70))
    wy1 = int(h * zone_top)
    wy2 = int(h * zone_bot)
    water = img.crop((0, wy1, w, wy2))
    ww, wh = water.size
    print(f"    Water zone: y={wy1}–{wy2} ({ww}x{wh}px)")

    # Compute tile dimensions
    tile_w = int(ww / (TILE_COLS - TILE_OVERLAP * (TILE_COLS - 1)))
    tile_h = int(wh / (TILE_ROWS - TILE_OVERLAP * (TILE_ROWS - 1)))
    step_x = int(tile_w * (1 - TILE_OVERLAP))
    step_y = int(tile_h * (1 - TILE_OVERLAP))
    total_tiles = TILE_COLS * TILE_ROWS
    print(f"    Tiles: {TILE_COLS}×{TILE_ROWS} ({total_tiles} total), size={tile_w}×{tile_h}, step={step_x}×{step_y}")

    model = YOLO("yolov8n.pt")
    all_dets = []  # (x1, y1, x2, y2, conf) in full frame coords

    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            tx1 = col * step_x
            ty1 = row * step_y
            tx2 = min(tx1 + tile_w, ww)
            ty2 = min(ty1 + tile_h, wh)

            tile = water.crop((tx1, ty1, tx2, ty2))
            tile = ImageEnhance.Contrast(tile).enhance(CONTRAST_BOOST)
            tile_np = np.array(tile)

            results = model(tile_np, classes=[0], conf=CONF_THRESHOLD, verbose=False)
            tile_dets = []
            for box in results[0].boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # Convert back to full frame coords
                all_dets.append((
                    bx1 + tx1,
                    by1 + ty1 + wy1,
                    bx2 + tx1,
                    by2 + ty1 + wy1,
                    conf
                ))
                tile_dets.append(conf)

            status = f"{len(tile_dets)} det(s) {[f'{c:.2f}' for c in tile_dets]}" if tile_dets else "—"
            print(f"    Tile ({row},{col}): {status}")

    print(f"\n    Raw detections: {len(all_dets)}")

    # Non-maximum suppression to remove duplicates from tile overlap
    sorted_dets = sorted(all_dets, key=lambda x: x[4], reverse=True)
    keep = []
    for det in sorted_dets:
        if all(iou(det, k) < NMS_IOU_THRESH for k in keep):
            keep.append(det)

    print(f"    After NMS dedup: {len(keep)}")
    return keep, img, (wy1, wy2)

# ─── ANNOTATE & SAVE ─────────────────────────────────────────────────────────

def save_annotated(img, detections, water_zone_y, spot_key, timestamp):
    wy1, wy2 = water_zone_y
    w, h = img.size
    full_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Draw water zone boundary
    cv2.rectangle(full_np, (0, wy1), (w, wy2), (0, 165, 255), 2)
    cv2.putText(full_np, "DETECTION ZONE", (12, wy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Draw each detection
    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        cv2.rectangle(full_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(full_np, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Count overlay
    count_text = f"SURFERS: {len(detections)}"
    cv2.rectangle(full_np, (0, 0), (260, 50), (0, 0, 0), -1)
    cv2.putText(full_np, count_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    ts_text = timestamp.strftime("%Y-%m-%d %H:%M UTC")
    cv2.putText(full_np, ts_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    out_path = OUTPUT_DIR / "frame_tiled_detection.jpg"
    cv2.imwrite(str(out_path), full_np)
    print(f"\n[5] Annotated frame → {out_path}")
    return out_path

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="Path to an existing frame jpg (skips fetch)")
    args = parser.parse_args()

    now = datetime.utcnow()
    print("=" * 55)
    print("SURF CROWD MONITOR — TILED DETECTION SPIKE v2")
    print(f"Spot:  {TEST_SPOT}")
    print(f"Time:  {now.isoformat()}Z")
    print("=" * 55)

    if args.local:
        frame_path = Path(args.local)
        print(f"\nUsing local frame: {frame_path}")
    else:
        frame_path = fetch_frame(TEST_SPOT)
        if not frame_path:
            sys.exit(1)

    detections, img, water_zone_y = run_tiled_detection(frame_path, TEST_SPOT)
    out_path = save_annotated(img, detections, water_zone_y, TEST_SPOT, now)

    # Summary
    print("\n" + "=" * 55)
    print("RESULT SUMMARY")
    print("=" * 55)
    summary = {
        "spot":       TEST_SPOT,
        "timestamp":  now.isoformat(),
        "surfer_count": len(detections),
        "mean_conf":  round(sum(d[4] for d in detections) / len(detections), 3) if detections else 0,
        "detections": [{"x1": round(d[0]), "y1": round(d[1]),
                        "x2": round(d[2]), "y2": round(d[3]),
                        "conf": round(d[4], 3)} for d in detections]
    }
    print(json.dumps(summary, indent=2))
    print(f"\n→ Open {out_path} to visually verify")
    print("=" * 55)
