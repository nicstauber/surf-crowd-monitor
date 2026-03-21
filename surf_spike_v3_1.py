"""
Surf Crowd Monitor — Spike v3.1
=================================
Recalibrated frame quality scorer.

Key fix: glare detection now uses texture-based analysis (Laplacian variance
+ patch noise) rather than pixel brightness thresholds. Diffuse afternoon
glare — which doesn't produce hot-spot pixels but does produce extreme water
surface noise — is now correctly identified as poor quality.

Calibrated against a known-bad afternoon glare frame from Lower Trestles:
  Laplacian variance:  5919  → POOR
  Noisy patch fraction: 85%  → POOR
  Expected morning values: Laplacian < 1500, noisy patches < 40%

Usage:
  python3 surf_spike_v3_1.py --local spike_output/frame.jpg
  python3 surf_spike_v3_1.py  (fetch fresh frame)
"""

import requests
import subprocess
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

WATER_ZONE = {
    "lower_trestles": (0.28, 0.65),
    "hb_pier_south":  (0.25, 0.65),
    "hb_cliffs":      (0.25, 0.65),
    "newport_56th":   (0.25, 0.65),
}

# ── Quality thresholds (calibrated from real frame data) ──────────────────

# Texture / glare
# Laplacian variance: measures high-frequency noise (sparkle, glare texture)
# Calibrated: bad afternoon frame = 5919. Clean morning target < 1500.
LAP_VAR_CLEAN  = 800    # below = very clean water, score 1.0
LAP_VAR_BAD    = 4000   # above = heavy glare, score 0.0

# Patch noise: % of 16px patches with std > 20 (noisy = glary)
# Calibrated: bad afternoon = 85%. Clean morning target < 35%.
PATCH_NOISE_CLEAN = 0.25   # below = clean, score 1.0
PATCH_NOISE_BAD   = 0.70   # above = heavy glare, score 0.0

# Brightness
BRIGHTNESS_MIN  = 60    # too dark
BRIGHTNESS_MAX  = 200   # too bright / overexposed
BRIGHTNESS_IDEAL = 115  # sweet spot

# Contrast (global std) — low = fog/dark/flat
CONTRAST_MIN_GOOD = 25

# Overall grade cutoffs
QUALITY_UNUSABLE = 0.30
QUALITY_POOR     = 0.45
QUALITY_MARGINAL = 0.62
QUALITY_GOOD     = 0.75

# Detection
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

# ─── STREAM FETCH ────────────────────────────────────────────────────────────

def fetch_frame(spot_key):
    url = SPOTS[spot_key]
    print(f"\n[1] Fetching: {spot_key}")
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"    ✗ {r.status_code}"); return None
    lines = [l.strip() for l in r.text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        print("    ✗ No segments"); return None
    seg_url = lines[-1] if lines[-1].startswith("http") else f"{url.rsplit('/',1)[0]}/{lines[-1]}"
    r2 = requests.get(seg_url, headers=HEADERS, timeout=30)
    if r2.status_code != 200:
        print(f"    ✗ Segment failed"); return None
    seg_path = OUTPUT_DIR / "segment.ts"
    seg_path.write_bytes(r2.content)
    frame_path = OUTPUT_DIR / "frame.jpg"
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(seg_path), "-frames:v", "1", "-q:v", "2", str(frame_path)],
        capture_output=True
    )
    if res.returncode != 0:
        print("    ✗ ffmpeg failed"); return None
    print(f"    ✓ Frame → {frame_path}")
    return frame_path

# ─── FRAME QUALITY SCORER (recalibrated) ─────────────────────────────────────

def score_frame_quality(img_pil, spot_key, save_debug=True):
    print(f"\n[QC] Scoring frame quality (v3.1 — texture-based glare)...")

    w, h = img_pil.size
    zone_top, zone_bot = WATER_ZONE.get(spot_key, (0.25, 0.70))
    wy1 = int(h * zone_top)
    wy2 = int(h * zone_bot)

    water_np   = np.array(img_pil.crop((0, wy1, w, wy2)))
    water_gray = cv2.cvtColor(water_np, cv2.COLOR_RGB2GRAY)

    notes = {}
    dims  = {}

    # ── 1. TEXTURE GLARE (Laplacian variance) ────────────────────────────────
    lap       = cv2.Laplacian(water_gray, cv2.CV_64F)
    lap_var   = float(lap.var())
    # Linear interpolation: LAP_VAR_CLEAN → 1.0, LAP_VAR_BAD → 0.0
    lap_score = 1.0 - min(1.0, max(0.0, (lap_var - LAP_VAR_CLEAN) / (LAP_VAR_BAD - LAP_VAR_CLEAN)))
    dims["texture_glare_lap"] = {
        "score":       round(lap_score, 3),
        "lap_var":     round(lap_var, 1),
        "clean_below": LAP_VAR_CLEAN,
        "bad_above":   LAP_VAR_BAD,
    }

    # ── 2. PATCH NOISE (% noisy 16px patches) ────────────────────────────────
    patch_stds = []
    for y in range(0, water_gray.shape[0] - 16, 16):
        for x in range(0, water_gray.shape[1] - 16, 16):
            patch_stds.append(float(water_gray[y:y+16, x:x+16].std()))
    noisy_frac  = float(np.mean(np.array(patch_stds) > 20))
    patch_score = 1.0 - min(1.0, max(0.0, (noisy_frac - PATCH_NOISE_CLEAN) / (PATCH_NOISE_BAD - PATCH_NOISE_CLEAN)))
    dims["texture_glare_patch"] = {
        "score":        round(patch_score, 3),
        "noisy_pct":    round(noisy_frac * 100, 1),
        "clean_below":  round(PATCH_NOISE_CLEAN * 100, 0),
        "bad_above":    round(PATCH_NOISE_BAD * 100, 0),
    }

    # Combined texture glare score (average of the two texture metrics)
    glare_score = (lap_score + patch_score) / 2.0
    dims["glare_combined"] = {"score": round(glare_score, 3)}

    # ── 3. BRIGHTNESS ────────────────────────────────────────────────────────
    mean_brightness = float(water_gray.mean())
    if mean_brightness < BRIGHTNESS_MIN:
        brightness_score = mean_brightness / BRIGHTNESS_MIN
    elif mean_brightness > BRIGHTNESS_MAX:
        brightness_score = max(0.0, 1.0 - (mean_brightness - BRIGHTNESS_MAX) / 55.0)
    else:
        brightness_score = 1.0 - abs(mean_brightness - BRIGHTNESS_IDEAL) / 80.0
    brightness_score = max(0.0, min(1.0, brightness_score))
    dims["brightness"] = {
        "score":      round(brightness_score, 3),
        "mean":       round(mean_brightness, 1),
        "ideal":      BRIGHTNESS_IDEAL,
    }

    # ── 4. CONTRAST ──────────────────────────────────────────────────────────
    std = float(water_gray.std())
    # Note: for glary frames std is actually high (lots of variation from sparkle)
    # So we only penalize for LOW contrast (fog/dark), not high
    contrast_score = min(1.0, std / CONTRAST_MIN_GOOD) if std < CONTRAST_MIN_GOOD else 1.0
    dims["contrast"] = {
        "score": round(contrast_score, 3),
        "std":   round(std, 1),
    }

    # ── OVERALL SCORE ─────────────────────────────────────────────────────────
    # Glare is dominant failure mode — weighted heavily
    weights = {
        "glare_combined": 0.55,
        "brightness":     0.25,
        "contrast":       0.20,
    }
    overall = sum(dims[k]["score"] * weights[k] for k in weights)
    overall = round(overall, 3)

    # Grade
    if overall >= QUALITY_GOOD:
        grade = "GOOD";     reliable = True;  symbol = "✓"
    elif overall >= QUALITY_MARGINAL:
        grade = "MARGINAL"; reliable = True;  symbol = "~"
    elif overall >= QUALITY_POOR:
        grade = "POOR";     reliable = False; symbol = "✗"
    else:
        grade = "UNUSABLE"; reliable = False; symbol = "✗✗"

    # Human notes
    note_lines = []
    if lap_var > LAP_VAR_BAD:
        note_lines.append(f"Heavy diffuse glare — water texture very noisy (Laplacian={lap_var:.0f}, threshold={LAP_VAR_BAD})")
    elif lap_var > LAP_VAR_CLEAN:
        note_lines.append(f"Moderate glare/texture noise (Laplacian={lap_var:.0f})")
    if noisy_frac > PATCH_NOISE_BAD:
        note_lines.append(f"Extreme patch noise — {noisy_frac*100:.0f}% of water zone patches are noisy")
    elif noisy_frac > PATCH_NOISE_CLEAN:
        note_lines.append(f"Elevated patch noise — {noisy_frac*100:.0f}% of water zone patches are noisy")
    if mean_brightness < BRIGHTNESS_MIN:
        note_lines.append(f"Frame too dark (brightness={mean_brightness:.0f})")
    elif mean_brightness > BRIGHTNESS_MAX:
        note_lines.append(f"Frame overexposed (brightness={mean_brightness:.0f})")
    if not note_lines:
        note_lines.append("Frame looks clean — good conditions for detection")
    if not reliable:
        note_lines.append(f"Count unreliable at this quality level [{grade}]")

    # Print summary
    print(f"    Laplacian var:     {lap_var:>8.1f}   score={lap_score:.2f}  (clean<{LAP_VAR_CLEAN}, bad>{LAP_VAR_BAD})")
    print(f"    Noisy patches:     {noisy_frac*100:>7.1f}%   score={patch_score:.2f}  (clean<{PATCH_NOISE_CLEAN*100:.0f}%, bad>{PATCH_NOISE_BAD*100:.0f}%)")
    print(f"    Glare combined:    {'':>9}   score={glare_score:.2f}")
    print(f"    Brightness:        {mean_brightness:>8.1f}   score={brightness_score:.2f}  (ideal={BRIGHTNESS_IDEAL})")
    print(f"    Contrast std:      {std:>8.1f}   score={contrast_score:.2f}")
    print(f"    {'─'*44}")
    print(f"    Overall quality:   {overall:.3f}   [{grade}]  {symbol}")
    for n in note_lines:
        print(f"    → {n}")

    # ── DEBUG FRAME ──────────────────────────────────────────────────────────
    if save_debug:
        debug = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Visualize Laplacian magnitude in water zone
        lap_abs    = np.abs(lap).astype(np.float32)
        lap_norm   = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lap_color  = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)  # blue=calm, red=noisy
        blended    = cv2.addWeighted(
            cv2.cvtColor(water_np, cv2.COLOR_RGB2BGR), 0.55,
            lap_color, 0.45, 0
        )
        debug[wy1:wy2, 0:w] = blended

        # Badge
        q_color = (0,200,0) if grade=="GOOD" else (0,165,255) if grade=="MARGINAL" else (0,0,220)
        cv2.rectangle(debug, (0,0), (680, 62), (0,0,0), -1)
        cv2.putText(debug, f"QUALITY: {overall:.2f}  [{grade}]  |  Lap={lap_var:.0f}  Noise={noisy_frac*100:.0f}%",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, q_color, 2)
        cv2.rectangle(debug, (0, wy1), (w, wy2), (0,165,255), 2)
        cv2.putText(debug, "SCORED ZONE  (color = texture noise: blue=calm  red=glary)",
                    (12, wy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        debug_path = OUTPUT_DIR / "frame_quality_debug.jpg"
        cv2.imwrite(str(debug_path), debug)
        print(f"    Debug → {debug_path}  (blue=calm water, red=glare/noise)")

    return {
        "overall_score": overall,
        "grade":         grade,
        "reliable":      reliable,
        "dimensions":    dims,
        "notes":         note_lines,
    }

# ─── DETECTION ───────────────────────────────────────────────────────────────

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
    wy1=int(h*zone_top); wy2=int(h*zone_bot)
    water=img.crop((0,wy1,w,wy2)); ww,wh=water.size
    tile_w=int(ww/(TILE_COLS-TILE_OVERLAP*(TILE_COLS-1)))
    tile_h=int(wh/(TILE_ROWS-TILE_OVERLAP*(TILE_ROWS-1)))
    step_x=int(tile_w*(1-TILE_OVERLAP)); step_y=int(tile_h*(1-TILE_OVERLAP))
    model=YOLO("yolov8n.pt")
    all_dets=[]
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            tx1=col*step_x; ty1=row*step_y
            tx2=min(tx1+tile_w,ww); ty2=min(ty1+tile_h,wh)
            tile=ImageEnhance.Contrast(water.crop((tx1,ty1,tx2,ty2))).enhance(CONTRAST_BOOST)
            results=model(np.array(tile),classes=[0],conf=CONF_THRESHOLD,verbose=False)
            for box in results[0].boxes:
                bx1,by1,bx2,by2=box.xyxy[0].tolist(); conf=float(box.conf[0])
                all_dets.append((bx1+tx1,by1+ty1+wy1,bx2+tx1,by2+ty1+wy1,conf))
    keep=[]
    for det in sorted(all_dets,key=lambda x:x[4],reverse=True):
        if all(iou(det,k)<NMS_IOU_THRESH for k in keep): keep.append(det)
    print(f"    Raw: {len(all_dets)}  After NMS: {len(keep)}")
    return keep, img, (wy1, wy2)

def save_annotated(img, detections, water_zone_y, quality, spot_key, ts):
    wy1,wy2=water_zone_y; w,h=img.size
    out=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    cv2.rectangle(out,(0,wy1),(w,wy2),(0,165,255),2)
    for (x1,y1,x2,y2,conf) in detections:
        cv2.rectangle(out,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(out,f"{conf:.2f}",(int(x1),int(y1)-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    grade=quality["grade"]; overall=quality["overall_score"]
    q_color=(0,200,0) if grade=="GOOD" else (0,165,255) if grade=="MARGINAL" else (0,0,220)
    cv2.rectangle(out,(0,0),(500,90),(0,0,0),-1)
    cv2.putText(out,f"SURFERS: {len(detections)}",(10,38),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,0),2)
    cv2.putText(out,f"FRAME QUALITY: {overall:.2f}  [{grade}]",(10,76),cv2.FONT_HERSHEY_SIMPLEX,0.7,q_color,2)
    cv2.putText(out,ts.strftime("%Y-%m-%d %H:%M UTC"),(10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
    out_path=OUTPUT_DIR/"frame_final.jpg"
    cv2.imwrite(str(out_path),out)
    return out_path

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="Use existing frame")
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()

    now = datetime.utcnow()
    print("=" * 60)
    print("SURF CROWD MONITOR — v3.1  (texture-based glare detection)")
    print(f"Spot: {TEST_SPOT}  |  {now.strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 60)

    frame_path = Path(args.local) if args.local else fetch_frame(TEST_SPOT)
    if not frame_path: sys.exit(1)

    img_pil = Image.open(frame_path).convert("RGB")
    quality = score_frame_quality(img_pil, TEST_SPOT)

    if args.quality_only: sys.exit(0)

    if not quality["reliable"]:
        print(f"\n⚠  [{quality['grade']}] — running detection anyway for spike validation")

    detections, img, water_zone_y = run_tiled_detection(frame_path, TEST_SPOT)
    out_path = save_annotated(img, detections, water_zone_y, quality, TEST_SPOT, now)

    record = {
        "spot":           TEST_SPOT,
        "timestamp":      now.isoformat(),
        "surfer_count":   len(detections),
        "count_reliable": quality["reliable"],
        "frame_quality": {
            "score":       quality["overall_score"],
            "grade":       quality["grade"],
            "lap_var":     quality["dimensions"]["texture_glare_lap"]["lap_var"],
            "noisy_pct":   quality["dimensions"]["texture_glare_patch"]["noisy_pct"],
            "brightness":  quality["dimensions"]["brightness"]["mean"],
        },
        "notes": quality["notes"],
    }

    print("\n" + "=" * 60)
    print("DB RECORD")
    print("=" * 60)
    print(json.dumps(record, indent=2))
    print(f"\n→ {out_path}")
    print(f"→ spike_output/frame_quality_debug.jpg  (blue=calm, red=noisy/glare)")
    print("=" * 60)
