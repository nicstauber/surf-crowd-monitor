"""
How close and how steady are the surfer counts? Old detector vs the current one.

Runs every frame in eval/frames.json through detect.py as it is at --base-ref
(default origin/main) and as it is in the working tree — once per --grid for
the current version — several times each. Reports, per version:
  - error against the hand count (`truth`, from the Lineup Answer Key page)
    for frames that have one,
  - how much the count moves between identical calls,
  - how often a count of 20+ lands on a multiple of 5.

Usage:
    python scripts/eval_counts.py                       # 3 runs, current grid
    python scripts/eval_counts.py --grids 3x2,4x2 --runs 2
    python scripts/eval_counts.py --base-ref HEAD~1

Each current-version run prints a POINTS line (frame, arm, run, points on a
0-1000 scale) so located surfers can be drawn back onto the frame; with
--overlay-dir the script draws the first run itself.

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
from statistics import mean

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
    ap.add_argument("--grids", help="comma-separated tile grids for the current version, e.g. 3x2,4x2")
    ap.add_argument("--full", action="store_true", help="also run the conditions call")
    ap.add_argument("--overlay-dir", type=Path)
    args = ap.parse_args()

    settings = json.loads((_ROOT / "config" / "settings.json").read_text())
    frames   = json.loads((_ROOT / "eval" / "frames.json").read_text())["frames"]
    grids    = args.grids.split(",") if args.grids else ["x".join(map(str, settings["tile_grid"]))]

    # arm label -> (module, settings)
    arms = {"base": (load_base_detect(args.base_ref), settings)}
    for g in grids:
        c, r = (int(v) for v in g.split("x"))
        arms[f"tiles {g}"] = (current_detect, {**settings, "tile_grid": [c, r]})

    client = anthropic.Anthropic()
    try:
        client.models.retrieve(settings["claude_model"])
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
            anthropic.NotFoundError, TypeError) as e:
        sys.exit(f"❌ Can't use the Claude API: {e}")

    images = [fetch(f["url"]) for f in frames]
    jobs   = [(i, a, r) for i in range(len(frames)) for a in arms for r in range(args.runs)]

    def run(job):
        i, a, r = job
        mod, arm_settings = arms[a]
        return job, mod.analyze_frame(images[i], f"{frames[i]['id']}/{a}/r{r}",
                                      client, arm_settings, include_conditions=args.full)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, jobs))

    counts = {(i, a): [] for i in range(len(frames)) for a in arms}
    tokens = {a: [] for a in arms}
    for (i, a, r), res in results:
        counts[(i, a)].append(res["count"])
        tokens[a].append(res.get("input_tokens", 0) + res.get("output_tokens", 0))
        if a != "base" and res["count"] >= 0:
            print(f"POINTS {frames[i]['id']} {a.replace(' ', '_')} r{r} "
                  f"{json.dumps(res.get('points', []), separators=(',', ':'))}")
            if args.overlay_dir and r == 0:
                args.overlay_dir.mkdir(parents=True, exist_ok=True)
                draw_points(images[i], res["points"],
                            args.overlay_dir / f"{frames[i]['id']}_{a.replace(' ', '_')}.jpg")

    names = list(arms)
    print(f"\n{'Frame':<6} {'Spot':<20} {'Truth':>5}  " + "  ".join(f"{a:<14}" for a in names))
    print("─" * (34 + 16 * len(names)))
    for i, f in enumerate(frames):
        truth = "—" if f.get("truth") is None else str(f["truth"])
        cells = "  ".join(f"{','.join(map(str, counts[(i, a)])):<14}" for a in names)
        print(f"{f['id']:<6} {f['spot']:<20} {truth:>5}  {cells}")

    print("\n" + "═" * 86)
    print(f"{'Version':<12} {'Avg error':>10} {'Bias':>6} {'Within ±2':>10} "
          f"{'Avg spread':>11} {'Round (≥20)':>12} {'Failed':>7} {'Tokens/frame':>12}")
    for a in names:
        errs = [c - f["truth"] for i, f in enumerate(frames)
                if f.get("truth") is not None and f["truth"] >= 0
                for c in counts[(i, a)] if c >= 0]
        valid   = [[c for c in counts[(i, a)] if c >= 0] for i in range(len(frames))]
        spreads = [max(c) - min(c) for c in valid if len(c) > 1]
        big     = [c for cs in valid for c in cs if c >= 20]
        round_  = f"{100 * sum(c % 5 == 0 for c in big) / len(big):.0f}%" if big else "—"
        failed  = sum(c < 0 for i in range(len(frames)) for c in counts[(i, a)])
        err_s   = f"{mean(abs(e) for e in errs):.1f}" if errs else "—"
        bias_s  = f"{mean(errs):+.1f}" if errs else "—"
        close_s = f"{100 * sum(abs(e) <= 2 for e in errs) / len(errs):.0f}%" if errs else "—"
        print(f"{a:<12} {err_s:>10} {bias_s:>6} {close_s:>10} "
              f"{mean(spreads) if spreads else 0:>11.1f} {round_:>12} {failed:>7} "
              f"{mean(tokens[a]):>12.0f}")
    print("═" * 86)
    print("Error and bias = count − hand count (bias < 0 means undercounting). Spread = max − min")
    print("across identical calls on one frame. Round = share of counts ≥20 that are multiples")
    print("of 5 (≈20% if nothing is rounded). Failed = calls that returned no usable count.")


if __name__ == "__main__":
    main()
