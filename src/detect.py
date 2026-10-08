"""
detect.py — Claude vision surfer counting + conditions assessment.

Sends a full-frame image to Claude and returns a structured surfer count
alongside a water conditions assessment in a single API call.
"""

import base64
import json
import logging
from io import BytesIO

import anthropic
from PIL import Image

log = logging.getLogger(__name__)

_PROMPT = """You are a precise surf cam analyst. Analyze this image and return two things: an exact surfer count and a conditions assessment.

PART 1 — SURFER COUNT
Count every person in the water using this method:
1. Scan LEFT to RIGHT in horizontal strips across the full water area
2. Mark every dark figure, dot, or silhouette on or in the water
3. Count each one individually — do not round or approximate

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

PART 2 — CONDITIONS ASSESSMENT
Assess the following fields using only the allowed values listed.

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
  "surfer_count": <exact integer>,
  "confidence": "<low|medium|high>",
  "count_notes": "<where surfers are located, e.g. 'cluster of 6 in left lineup, 3 scattered middle'>",
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
}

If truly unable to count due to darkness or glare, set surfer_count to -1."""


def _encode_image(img_pil: Image.Image, max_width: int) -> str:
    """Resize and base64-encode image as JPEG."""
    if img_pil.width > max_width:
        ratio   = max_width / img_pil.width
        img_pil = img_pil.resize(
            (max_width, int(img_pil.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def analyze_frame(
    img_pil: Image.Image,
    frame_label: str,
    client: anthropic.Anthropic,
    settings: dict,
) -> dict:
    """
    Send full frame to Claude for surfer count + conditions assessment.
    Returns a dict with keys: count, confidence, notes, conditions, conditions_notes,
    plus model / input_tokens / output_tokens when the API call succeeded.
    count is -1 if Claude cannot determine a reliable count.
    """
    b64   = _encode_image(img_pil, settings["image_width"])
    model = settings["claude_model"]

    # Haiku 5.5+ thinks by default and counts thinking toward max_tokens, so
    # the cap must leave room for it on top of the ~200-token JSON reply.
    kwargs = {}
    if settings.get("claude_effort"):
        kwargs["output_config"] = {"effort": settings["claude_effort"]}

    try:
        response = client.messages.create(
            model=model,
            max_tokens=settings.get("claude_max_tokens", 2000),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       b64,
                        },
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }],
            **kwargs,
        )

        usage = {
            "model":         model,
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        if response.stop_reason in ("refusal", "max_tokens"):
            log.warning(f"[{frame_label}] Claude stopped early: {response.stop_reason}")
            return {"count": -1, "confidence": "error",
                    "notes": f"stop_reason={response.stop_reason}",
                    "conditions": {}, "conditions_notes": "", **usage}

        # Newer models can lead with thinking blocks — read the text block by type.
        raw    = next(b.text for b in response.content if b.type == "text").strip()
        clean  = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)

        count      = int(parsed.get("surfer_count", -1))
        conf       = parsed.get("confidence", "unknown")
        notes      = parsed.get("count_notes", "")
        conditions = parsed.get("conditions", {})
        cond_notes = parsed.get("conditions_notes", "")

        log.info(f"[{frame_label}] Claude → count={count}  conf={conf}")
        log.info(f"[{frame_label}] {notes}")
        log.info(f"[{frame_label}] {cond_notes}")

        return {
            "count":            count,
            "confidence":       conf,
            "notes":            notes,
            "conditions":       conditions,
            "conditions_notes": cond_notes,
            **usage,
        }

    except json.JSONDecodeError as e:
        log.warning(f"[{frame_label}] JSON parse error: {e}")
        return {"count": -1, "confidence": "error", "notes": f"parse error: {e}",
                "conditions": {}, "conditions_notes": "", **usage}
    except Exception as e:
        log.warning(f"[{frame_label}] API error: {e}")
        return {"count": -1, "confidence": "error", "notes": str(e),
                "conditions": {}, "conditions_notes": ""}
