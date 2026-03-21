"""
detect.py — Claude vision surfer counting.

Sends a full-frame image to Claude and returns a structured count
with confidence level and descriptive notes.
"""

import base64
import json
import logging
from io import BytesIO

import anthropic
from PIL import Image

log = logging.getLogger(__name__)

_PROMPT = """You are a precise surf cam analyst. Your job is to count every single person in the water — not estimate, but count each one individually.

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


def _encode_image(img_pil: Image.Image, max_width: int) -> str:
    """Resize and base64-encode image as JPEG."""
    if img_pil.width > max_width:
        ratio   = max_width / img_pil.width
        img_pil = img_pil.resize(
            (max_width, int(img_pil.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def count_surfers(
    img_pil: Image.Image,
    frame_label: str,
    client: anthropic.Anthropic,
    settings: dict,
) -> tuple[int, str, str]:
    """
    Send full frame to Claude and return (count, confidence, notes).
    count is -1 if Claude cannot determine a reliable count.
    """
    b64   = _encode_image(img_pil, settings["image_width"])
    model = settings["claude_model"]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=400,
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
        )

        raw    = response.content[0].text.strip()
        clean  = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)

        count = int(parsed.get("count", -1))
        conf  = parsed.get("confidence", "unknown")
        notes = parsed.get("notes", "")

        log.info(f"[{frame_label}] Claude → count={count}  conf={conf}")
        log.info(f"[{frame_label}] {notes}")

        return count, conf, notes

    except json.JSONDecodeError as e:
        log.warning(f"[{frame_label}] JSON parse error: {e}")
        return -1, "error", f"parse error: {e}"
    except Exception as e:
        log.warning(f"[{frame_label}] API error: {e}")
        return -1, "error", str(e)
