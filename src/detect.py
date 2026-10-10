"""
detect.py — Claude vision surfer counting + conditions assessment.

The count comes from zoomed tiles: the frame is cut into a grid, each tile is
upscaled and sent to Claude, which returns a point per person in the water,
and the count is the number of points across all tiles. Asked for a total on
the whole frame, Haiku guessed instead of counting (Haiku 4.5 wrote exactly
47 for 731 crowded lineups; 5.5 rounded big crowds to multiples of 5), and a
full-frame point list put dots on empty water. Small, zoomed groups are what
it counts well.

Conditions (surface, swell size, ...) need the whole scene, so they come from
one separate full-frame call when include_conditions is set.
"""

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import anthropic
from PIL import Image

log = logging.getLogger(__name__)

_TILE_PROMPT = """You are a precise surf cam analyst. This image is one zoomed-in section of a wider surf cam frame, so it may show only sky, only sand, or part of the water.

Locate every person in the water and give each one its own point. The count is the number of points you list, so list each person exactly once and never estimate.
1. Scan LEFT to RIGHT in horizontal strips across the whole image
2. For every person on or in the water, add one [x, y] point at their centre. x and y are integers from 0 to 1000, measured from the image's left edge and top edge as a fraction of its width and height
3. A person cut off by the image edge counts only if most of their body is inside this image

COUNT THESE:
- Surfers sitting on boards in the lineup
- Surfers lying prone paddling
- Surfers actively riding a wave
- Anyone standing, wading, or swimming in the water

DO NOT COUNT:
- People on dry beach, sand, piers, jetties, rocks, or structures
- Umbrellas, towels, tents, or beach gear
- Birds, animals, or buoys
- Whitewash or foam that resembles a person

Respond ONLY with JSON, no other text:
{
  "obscured": <true|false>,
  "surfers": [[<x>, <y>], ...]
}

Set obscured to true only if darkness, fog, haze, or glare hides the water in this section so you cannot tell whether anyone is there. A section with no water in it, or with clearly visible empty water, is not obscured: return an empty surfers list."""


_CONDITIONS_PROMPT = """You are a precise surf cam analyst. Assess the water conditions in this surf cam image using only the allowed values listed.

surface: texture of the open water surface
  glassy | light_chop | choppy | very_choppy

swell_size: estimated wave face height
  flat | ankle | knee | waist | chest | head | over_head

wave_quality: shape and form of breaking waves
  clean | crumbly | mushy | closed_out

wind_effect: infer from surface texture and spray off wave lips
  offshore | onshore | cross_shore | calm

crowd_distribution: how surfers are positioned in the lineup
  empty | spread | clustered | multiple_peaks

water_clarity: color and turbidity of the water
  clear | murky | brown

lighting: current light quality in the scene
  golden_hour | overcast_flat | harsh_midday | backlit

visibility: atmospheric clarity toward the horizon
  clear | hazy | foggy

IMPORTANT: Use only the exact values listed for each field. Do not invent new values.

Respond ONLY with JSON, no other text:
{
  "conditions": {
    "surface": "<value>",
    "swell_size": "<value>",
    "wave_quality": "<value>",
    "wind_effect": "<value>",
    "crowd_distribution": "<value>",
    "water_clarity": "<value>",
    "lighting": "<value>",
    "visibility": "<value>"
  },
  "conditions_notes": "<one sentence describing the overall session — e.g. 'Small clean chest-high sets with offshore grooming, glassy surface, light crowd spread across the peak'>"
}"""


def _dedupe_points(raw: list, min_gap: int = 3) -> list[list[int]]:
    """
    Clean Claude's [x, y] list (0-1000 scale): drop malformed or out-of-range
    entries, and merge points closer than min_gap on both axes,
    which are the same person listed twice rather than two people.
    """
    kept: list[list[int]] = []
    for p in raw:
        try:
            x, y = (int(round(float(v))) for v in p)
        except (TypeError, ValueError):
            continue
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            continue
        if any(abs(x - kx) < min_gap and abs(y - ky) < min_gap for kx, ky in kept):
            continue
        kept.append([x, y])
    return kept


def _encode_image(img_pil: Image.Image, width: int, upscale: bool = False) -> str:
    """Resize to `width` (only downwards unless upscale) and base64-encode as JPEG."""
    if img_pil.width > width or (upscale and img_pil.width < width):
        ratio   = width / img_pil.width
        img_pil = img_pil.resize((width, int(img_pil.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _ask(client, settings, b64: str, prompt: str) -> tuple[dict, dict]:
    """One vision call. Returns (parsed JSON, usage); raises on any failure."""
    model = settings["claude_model"]
    # Haiku 5.5+ thinks by default and counts thinking toward max_tokens, so
    # the cap must leave room for it on top of the JSON reply.
    kwargs = {}
    if settings.get("claude_effort"):
        kwargs["output_config"] = {"effort": settings["claude_effort"]}
    response = client.messages.create(
        model=model,
        max_tokens=settings.get("claude_max_tokens", 4000),
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
        **kwargs,
    )
    usage = {"input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    # Newer models can lead with thinking blocks — read the text block by type.
    raw = next(b.text for b in response.content if b.type == "text").strip()
    return json.loads(raw.replace("```json", "").replace("```", "").strip()), usage


def _tiles(img: Image.Image, cols: int, rows: int):
    """Yield (left, top, width, height) boxes of a cols x rows grid."""
    for r in range(rows):
        for c in range(cols):
            left, top = img.width * c // cols, img.height * r // rows
            right, bottom = img.width * (c + 1) // cols, img.height * (r + 1) // rows
            yield left, top, right - left, bottom - top


def count_surfers(img_pil: Image.Image, frame_label: str, client, settings: dict) -> dict:
    """
    Count people in the water tile by tile. Returns count (-1 when it can't be
    trusted), points in whole-frame 0-1000 coordinates, notes, and token usage.
    """
    cols, rows = settings.get("tile_grid", [3, 2])
    width      = settings.get("tile_width", 1024)
    boxes      = list(_tiles(img_pil, cols, rows))

    def one(box):
        left, top, w, h = box
        tile = img_pil.crop((left, top, left + w, top + h))
        parsed, usage = _ask(client, settings, _encode_image(tile, width, upscale=True), _TILE_PROMPT)
        if not isinstance(parsed.get("surfers"), list):
            raise ValueError("reply has no surfers list")
        pts = [[round((left + x * w / 1000) * 1000 / img_pil.width),
                 round((top + y * h / 1000) * 1000 / img_pil.height)]
               for x, y in _dedupe_points(parsed["surfers"])]
        return bool(parsed.get("obscured")), pts, usage

    in_tok = out_tok = 0
    try:
        with ThreadPoolExecutor(max_workers=len(boxes)) as pool:
            results = list(pool.map(one, boxes))
    except Exception as e:
        log.warning(f"[{frame_label}] tile call failed: {e}")
        return {"count": -1, "points": [], "confidence": "error", "notes": str(e),
                "input_tokens": 0, "output_tokens": 0}

    points, obscured = [], 0
    for is_obscured, pts, usage in results:
        obscured += is_obscured
        points   += pts
        in_tok   += usage["input_tokens"]
        out_tok  += usage["output_tokens"]

    # Sky and sand tiles are clear and empty, so one hidden tile can still be
    # most of the water. Trust the count only when no tile was hidden.
    count = -1 if obscured else len(points)
    notes = f"{len(points)} located in {len(boxes)} tiles ({cols}x{rows})"
    if obscured:
        notes += f"; {obscured} tile(s) obscured"
    log.info(f"[{frame_label}] Claude tiles → count={count}  {notes}")
    return {"count": count, "points": points if count >= 0 else [],
            "confidence": "low" if obscured else "high", "notes": notes,
            "input_tokens": in_tok, "output_tokens": out_tok}


def analyze_frame(
    img_pil: Image.Image,
    frame_label: str,
    client: anthropic.Anthropic,
    settings: dict,
    include_conditions: bool = True,
) -> dict:
    """
    Count surfers (tiled) and, when include_conditions is set, assess
    conditions from one full-frame call.
    Returns a dict with keys: count, points, confidence, notes, conditions,
    conditions_notes, model, input_tokens, output_tokens.
    count is -1 if Claude cannot determine a reliable count.
    """
    result = count_surfers(img_pil, frame_label, client, settings)
    result.update({"model": settings["claude_model"], "conditions": {}, "conditions_notes": ""})

    if include_conditions:
        try:
            parsed, usage = _ask(client, settings,
                                 _encode_image(img_pil, settings["image_width"]), _CONDITIONS_PROMPT)
            result["conditions"]       = parsed.get("conditions", {})
            result["conditions_notes"] = parsed.get("conditions_notes", "")
            result["input_tokens"]    += usage["input_tokens"]
            result["output_tokens"]   += usage["output_tokens"]
            log.info(f"[{frame_label}] {result['conditions_notes']}")
        except Exception as e:
            # A missed assessment leaves conditions empty; the count still stands.
            log.warning(f"[{frame_label}] conditions call failed: {e}")
    return result
