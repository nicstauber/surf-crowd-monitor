"""
Surf Crowd Monitor — Detection Spike
=====================================
Tests:
  1. Can we grab a frame from a Surfline HLS stream?
  2. Can we run person detection on it?
  3. What does the raw count look like?

Requirements:
  pip install requests opencv-python ultralytics pillow

Usage:
  python surf_spike.py
"""

import requests
import subprocess
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Spot slugs found in the HLS CDN path — update these to test other cams
SPOTS = {
    "lower_trestles":    "https://hls.cdn-surfline.com/oregon/wc-lowerslefts/playlist.m3u8",
    "hb_pier_south":     "https://hls.cdn-surfline.com/oregon/wc-huntingtonbeachsouthside/playlist.m3u8",  # slug TBD
    "hb_cliffs":         "https://hls.cdn-surfline.com/oregon/wc-huntingtoncliffs/playlist.m3u8",           # slug TBD
    "newport_56th":      "https://hls.cdn-surfline.com/oregon/wc-56thstreet/playlist.m3u8",                 # slug TBD
}

# Start with just Trestles for the spike
TEST_SPOT = "lower_trestles"
TEST_URL  = SPOTS[TEST_SPOT]

# Mimics the browser headers seen in the Network tab — no auth token needed
HEADERS = {
    "origin":       "https://www.surfline.com",
    "referer":      "https://www.surfline.com/",
    "user-agent":   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "accept":       "*/*",
}

OUTPUT_DIR = Path("./spike_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── STEP 1: FETCH PLAYLIST ──────────────────────────────────────────────────
def fetch_playlist(url):
    print(f"\n[1] Fetching HLS playlist...")
    print(f"    URL: {url}")
    r = requests.get(url, headers=HEADERS, timeout=15)
    print(f"    Status: {r.status_code}")
    if r.status_code != 200:
        print(f"    ✗ Failed. Response: {r.text[:300]}")
        return None
    print(f"    ✓ Got playlist ({len(r.text)} bytes)")
    print(f"    Preview:\n{r.text[:400]}\n")
    return r.text

# ─── STEP 2: PARSE SEGMENT URL FROM PLAYLIST ─────────────────────────────────
def get_latest_segment(playlist_text, base_url):
    """
    HLS playlists list .ts segment filenames.
    We grab the last one (most recent video segment).
    """
    lines = [l.strip() for l in playlist_text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        print("    ✗ No segments found in playlist")
        return None

    latest = lines[-1]
    # Segments may be relative or absolute URLs
    if latest.startswith("http"):
        segment_url = latest
    else:
        base = base_url.rsplit("/", 1)[0]
        segment_url = f"{base}/{latest}"

    print(f"[2] Latest segment: {latest}")
    print(f"    Full URL: {segment_url}")
    return segment_url

# ─── STEP 3: DOWNLOAD SEGMENT ────────────────────────────────────────────────
def download_segment(segment_url):
    print(f"\n[3] Downloading segment...")
    r = requests.get(segment_url, headers=HEADERS, timeout=30)
    print(f"    Status: {r.status_code}")
    if r.status_code != 200:
        print(f"    ✗ Failed")
        return None
    segment_path = OUTPUT_DIR / "segment.ts"
    segment_path.write_bytes(r.content)
    print(f"    ✓ Saved {len(r.content)/1024:.1f} KB → {segment_path}")
    return segment_path

# ─── STEP 4: EXTRACT FRAME FROM SEGMENT ─────────────────────────────────────
def extract_frame(segment_path):
    """
    Use ffmpeg to pull a single frame from the .ts video segment.
    ffmpeg must be installed: brew install ffmpeg
    """
    print(f"\n[4] Extracting frame with ffmpeg...")
    frame_path = OUTPUT_DIR / "frame.jpg"

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(segment_path),
        "-frames:v", "1",          # grab exactly one frame
        "-q:v", "2",               # high quality JPEG
        str(frame_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ✗ ffmpeg error:\n{result.stderr[-500:]}")
        print(f"    Make sure ffmpeg is installed: brew install ffmpeg")
        return None

    print(f"    ✓ Frame saved → {frame_path}")
    return frame_path

# ─── STEP 5: RUN YOLO DETECTION ──────────────────────────────────────────────
def run_detection(frame_path):
    """
    Runs YOLOv8n (nano — fastest, good enough for spike) person detection.
    Saves an annotated version of the frame with bounding boxes.
    """
    print(f"\n[5] Running YOLOv8 person detection...")
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("    ✗ Missing packages. Run: pip install ultralytics opencv-python")
        return None

    # Load model (downloads ~6MB on first run)
    model = YOLO("yolov8n.pt")

    # Run inference — class 0 = person
    results = model(str(frame_path), classes=[0], conf=0.3, verbose=False)
    result  = results[0]

    count       = len(result.boxes)
    confidences = [float(b.conf) for b in result.boxes] if count > 0 else []
    mean_conf   = sum(confidences) / len(confidences) if confidences else 0

    print(f"    ✓ Detections: {count} person(s)")
    print(f"    Confidences: {[f'{c:.2f}' for c in confidences]}")
    print(f"    Mean confidence: {mean_conf:.2f}")

    # Save annotated frame
    annotated_path = OUTPUT_DIR / "frame_annotated.jpg"
    annotated = result.plot()
    import cv2
    cv2.imwrite(str(annotated_path), annotated)
    print(f"    ✓ Annotated frame → {annotated_path}")

    return {
        "spot":        TEST_SPOT,
        "timestamp":   datetime.utcnow().isoformat(),
        "count":       count,
        "confidences": confidences,
        "mean_conf":   round(mean_conf, 3),
        "frame":       str(annotated_path),
    }

# ─── STEP 6: PRINT RESULT SUMMARY ────────────────────────────────────────────
def print_summary(result):
    print("\n" + "="*50)
    print("SPIKE RESULT SUMMARY")
    print("="*50)
    if result:
        print(json.dumps(result, indent=2))
        print("\n✓ Open spike_output/frame_annotated.jpg to visually verify detections")
        print("✓ Open spike_output/frame.jpg to see the raw captured frame")
    else:
        print("✗ Spike did not complete — check errors above")
    print("="*50)

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("SURF CROWD MONITOR — DETECTION SPIKE")
    print(f"Spot: {TEST_SPOT}")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print("="*50)

    playlist = fetch_playlist(TEST_URL)
    if not playlist:
        print("\n✗ Could not fetch playlist. Possible reasons:")
        print("  - Surfline CDN requires auth (we'll need to log in first)")
        print("  - Wrong cam slug in URL")
        print("  - Cam is offline")
        sys.exit(1)

    segment_url = get_latest_segment(playlist, TEST_URL)
    if not segment_url:
        sys.exit(1)

    segment_path = download_segment(segment_url)
    if not segment_path:
        sys.exit(1)

    frame_path = extract_frame(segment_path)
    if not frame_path:
        sys.exit(1)

    detection_result = run_detection(frame_path)
    print_summary(detection_result)
