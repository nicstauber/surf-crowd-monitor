"""
Do surfer counts hold steady? Old prompt vs the current one, same frames.

Runs every frame in eval/frames.json through two versions of detect.py —
the one at --base-ref (default origin/main) and the one in the working tree —
several times each, and reports how much each version's count moves between
identical calls and how often it lands on a round number. A trustworthy
counter gives (nearly) the same answer every time and has no favourite numbers.

Usage:
    python scripts/eval_counts.py                   # 3 runs per frame per version
    python scripts/eval_counts.py --runs 5 --full   # full prompt incl. conditions
    python scripts/eval_counts.py --base-ref HEAD~1

Each current-version run prints a POINTS line (frame, run, [[x, y], ...] on a
0-1000 scale) so the located surfers can be drawn back onto the frame. With
--overlay-dir the script draws them itself.

Needs ANTHROPIC_API_KEY. Nothing is written to Supabase.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from statistics import mean, median

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import anthropic
from PIL import Image, ImageDraw

import detect as current_detect


def load_base_detect(ref: str):
    """Import detect.py as it was at a git ref, as a separate module."""
    src = subprocess.run(["git", "show", f"{ref}:src/detect.py"], cwd=_ROOT,
                         check=True, capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "detect_base.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("detect_base", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=30) as r:
        return Image.open(BytesIO(r.read())).convert("RGB")


def draw_points(img: Image.Image, points: list, out: Path) -> None:
    img = img.copy()
    d   = ImageDraw.Draw(img)
    r   = max(2, img.width // 200)
    for x, y in points:
        px, py = x * img.width / 1000, y * img.height / 1000
        d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 80), width=2)
    d.text((6, 6), f"{len(points)} located", fill=(255, 0, 80))
    img.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--base-ref", default="origin/main")
    ap.add_argument("--full", action="store_true", help="use the full conditions prompt")
    ap.add_argument("--overlay-dir", type=Path)
    args = ap.parse_args()

    settings = json.loads((_ROOT / "config" / "settings.json").read_text())
    frames   = json.loads((_ROOT / "eval" / "frames.json").read_text())["frames"]
    versions = {"base": load_base_detect(args.base_ref), "current": current_detect}
    client   = anthropic.Anthropic()
    try:
        client.models.retrieve(settings["claude_model"])
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
            anthropic.NotFoundError, TypeError) as e:
        sys.exit(f"❌ Can't use the Claude API: {e}")

    images = [fetch(f["url"]) for f in frames]
    jobs   = [(i, v, r) for i in range(len(frames)) for v in versions for r in range(args.runs)]

    def run(job):
        i, v, r = job
        res = versions[v].analyze_frame(images[i], f"{frames[i]['spot']}/{v}/r{r}",
                                        client, settings, include_conditions=args.full)
        return job, res

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, jobs))

    counts = {(i, v): [] for i in range(len(frames)) for v in versions}
    for (i, v, r), res in results:
        counts[(i, v)].append(res["count"])
        if v == "current" and res["count"] >= 0:
            print(f"POINTS {i:02d} r{r} {json.dumps(res.get('points', []), separators=(',', ':'))}")
            if args.overlay_dir and r == 0:
                args.overlay_dir.mkdir(parents=True, exist_ok=True)
                draw_points(images[i], res["points"],
                            args.overlay_dir / f"{i:02d}_{frames[i]['spot']}.jpg")

    print(f"\n{'#':>2} {'Spot':<20} {'Stored':>6}  {'Base runs':<16} {'Current runs':<16}")
    print("─" * 66)
    for i, f in enumerate(frames):
        fmt = lambda v: ",".join(str(c) for c in counts[(i, v)])
        print(f"{i:02d} {f['spot']:<20} {f['stored_count']:>6}  {fmt('base'):<16} {fmt('current'):<16}")

    print("\n" + "═" * 66)
    print(f"{'Version':<10} {'Avg spread':>11} {'Max spread':>11} {'Round (≥20)':>12} {'Failed':>7}")
    for v in versions:
        valid   = [[c for c in counts[(i, v)] if c >= 0] for i in range(len(frames))]
        spreads = [max(c) - min(c) for c in valid if len(c) > 1]
        big     = [c for cs in valid for c in cs if c >= 20]
        round_  = f"{100 * sum(c % 5 == 0 for c in big) / len(big):.0f}%" if big else "—"
        failed  = sum(c < 0 for i in range(len(frames)) for c in counts[(i, v)])
        print(f"{v:<10} {mean(spreads) if spreads else 0:>11.1f} {max(spreads, default=0):>11} "
              f"{round_:>12} {failed:>7}")
    print("═" * 66)
    print("Spread = max − min across identical calls on one frame. Round = share of")
    print("counts ≥20 that are multiples of 5 (≈20% if nothing is being rounded).")
    print(f"Medians — base: {[median(c) if c else None for c in ([x for x in counts[(i, 'base')] if x >= 0] for i in range(len(frames)))]}")
    print(f"Medians — current: {[median(c) if c else None for c in ([x for x in counts[(i, 'current')] if x >= 0] for i in range(len(frames)))]}")


if __name__ == "__main__":
    main()
