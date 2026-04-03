"""
capture.py — Frame fetch and quality scoring.

Fetches HLS stream frames for a given spot and scores each frame for
usability (glare, brightness, contrast) before sending to Claude.
"""

import requests
import time
import logging
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

log = logging.getLogger(__name__)

HLS_BASE = "https://hls.cdn-surfline.com/oregon"
HEADERS = {
    "origin":     "https://www.surfline.com",
    "referer":    "https://www.surfline.com/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "accept":     "*/*",
}

# Quality thresholds (fixed — not user-tunable)
_LAP_VAR_CLEAN     = 800
_LAP_VAR_BAD       = 4000
_PATCH_NOISE_CLEAN = 0.25
_PATCH_NOISE_BAD   = 0.70
_BRIGHTNESS_MIN    = 60
_BRIGHTNESS_MAX    = 200
_BRIGHTNESS_IDEAL  = 115


def _fetch_hls_frame(spot: dict, frame_index: int, output_dir: Path) -> Path:
    """Fetch one frame from the spot's HLS stream. Returns path to JPEG."""
    slug = spot["hls_slug"]
    url  = f"{HLS_BASE}/{slug}/playlist.m3u8"

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    lines = [l.strip() for l in r.text.splitlines()
             if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError(f"Empty playlist for {spot['id']}")

    last = lines[-1]
    seg_url = last if last.startswith("http") else f"{url.rsplit('/', 1)[0]}/{last}"

    r2 = requests.get(seg_url, headers=HEADERS, timeout=30)
    r2.raise_for_status()

    seg_path   = output_dir / f"{spot['id']}_seg_{frame_index}.ts"
    frame_path = output_dir / f"{spot['id']}_frame_{frame_index}.jpg"
    seg_path.write_bytes(r2.content)

    cap = cv2.VideoCapture(str(seg_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"OpenCV failed to read frame for {spot['id']} frame {frame_index}")
    cv2.imwrite(str(frame_path), frame)

    return frame_path


def fetch_burst(spot: dict, settings: dict, output_dir: Path):
    """
    Fetch a burst of frames from the spot's HLS stream.
    Returns list of frame paths (may be shorter than burst_frame_count on errors).
    """
    count    = settings["burst_frame_count"]
    interval = settings["burst_interval_seconds"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[{spot['id']}] Fetching {count}-frame burst ({interval}s intervals)")
    frames = []

    for i in range(count):
        if i > 0:
            time.sleep(interval)
        try:
            path = _fetch_hls_frame(spot, i + 1, output_dir)
            frames.append(path)
            log.info(f"[{spot['id']}]   Frame {i+1}/{count} ✓")
        except Exception as e:
            log.warning(f"[{spot['id']}]   Frame {i+1}/{count} failed: {e}")

    return frames


def score_frame_quality(img_pil: Image.Image, quality_threshold: float) -> dict:
    """
    Score overall frame quality for glare, brightness, and contrast.

    Samples the middle vertical band of the frame (25%–70%) as a proxy
    for the water zone — avoids sky at top and sand at bottom skewing metrics.

    Returns a dict with overall_score, grade, and raw metrics.
    """
    w, h   = img_pil.size
    sample = np.array(img_pil.crop((0, int(h * 0.25), w, int(h * 0.70))))
    gray   = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)

    # Glare: Laplacian variance + patch noise
    lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_score = 1.0 - min(1.0, max(0.0,
        (lap_var - _LAP_VAR_CLEAN) / (_LAP_VAR_BAD - _LAP_VAR_CLEAN)))

    patch_stds  = [float(gray[y:y+16, x:x+16].std())
                   for y in range(0, gray.shape[0] - 16, 16)
                   for x in range(0, gray.shape[1] - 16, 16)]
    noisy_frac  = float(np.mean(np.array(patch_stds) > 20))
    patch_score = 1.0 - min(1.0, max(0.0,
        (noisy_frac - _PATCH_NOISE_CLEAN) / (_PATCH_NOISE_BAD - _PATCH_NOISE_CLEAN)))
    glare_score = (lap_score + patch_score) / 2.0

    # Brightness
    mean_b = float(gray.mean())
    if mean_b < _BRIGHTNESS_MIN:
        b_score = mean_b / _BRIGHTNESS_MIN
    elif mean_b > _BRIGHTNESS_MAX:
        b_score = max(0.0, 1.0 - (mean_b - _BRIGHTNESS_MAX) / 55.0)
    else:
        b_score = 1.0 - abs(mean_b - _BRIGHTNESS_IDEAL) / 80.0
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
        "contributes":     overall >= quality_threshold,
        "lap_var":         round(lap_var, 1),
        "noisy_pct":       round(noisy_frac * 100, 1),
        "mean_brightness": round(mean_b, 1),
    }


def score_frames(frame_paths, settings: dict):
    """
    Load and score quality for all frames.
    Returns a list of dicts with keys: index, path, image, quality.
    """
    threshold = settings["quality_threshold"]
    results   = []

    for i, fp in enumerate(frame_paths):
        img_pil = Image.open(fp).convert("RGB")
        quality = score_frame_quality(img_pil, threshold)
        log.info(
            f"[frame {i+1}] quality={quality['overall_score']:.2f} [{quality['grade']}]  "
            f"lap={quality['lap_var']:.0f}  noise={quality['noisy_pct']:.0f}%  "
            f"bright={quality['mean_brightness']:.0f}"
        )
        results.append({
            "index":   i + 1,
            "path":    str(fp),
            "image":   img_pil,
            "quality": quality,
        })

    return results
