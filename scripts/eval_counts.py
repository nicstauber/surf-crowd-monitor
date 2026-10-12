"""
How close and how steady are the surfer counts? Old detector vs the current one.

Runs every frame in eval/frames.json through detect.py as it is at --base-ref
(default origin/main) and as it is in the working tree, several times each.
Reports, per version:
  - error against the hand count (`truth`, from the Lineup Answer Key page)
    for frames that have one,
  - how much the count moves between identical calls,
  - how often a count of 20+ lands on a multiple of 5.

Usage:
    python scripts/eval_counts.py                 # 3 runs per frame per version
    python scripts/eval_counts.py --runs 5 --full # include the conditions prompt
    python scripts/eval_counts.py --base-ref HEAD~1

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
from PIL import Image

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--base-ref", default="origin/main")
    ap.add_argument("--full", action="store_true", help="use the full conditions prompt")
    args = ap.parse_args()

    settings = json.loads((_ROOT / "config" / "settings.json").read_text())
    frames   = json.loads((_ROOT / "eval" / "frames.json").read_text())["frames"]

    # arm label -> (module, settings)
    arms = {"base": (load_base_detect(args.base_ref), settings),
            "current": (current_detect, settings)}

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
    ranges = {(i, a): [] for i in range(len(frames)) for a in arms}  # (sure, sure+maybe)
    tokens = {a: [] for a in arms}
    for (i, a, r), res in results:
        counts[(i, a)].append(res["count"])
        # A version without a range reports one number: a zero-width range.
        ranges[(i, a)].append((res.get("count_min", res["count"]), res.get("count_max", res["count"])))
        tokens[a].append(res.get("input_tokens", 0) + res.get("output_tokens", 0))

    def cell(i, a):
        return ",".join(str(c) if lo == hi else f"{lo}-{hi}"
                        for c, (lo, hi) in zip(counts[(i, a)], ranges[(i, a)]))

    names = list(arms)
    print(f"\n{'Frame':<6} {'Spot':<20} {'Truth':>5}  " + "  ".join(f"{a:<22}" for a in names))
    print("─" * (34 + 24 * len(names)))
    for i, f in enumerate(frames):
        truth = "—" if f.get("truth") is None else str(f["truth"])
        cells = "  ".join(f"{cell(i, a):<22}" for a in names)
        print(f"{f['id']:<6} {f['spot']:<20} {truth:>5}  {cells}")

    print("\n" + "═" * 108)
    print(f"{'Version':<12} {'Avg error':>10} {'Bias':>6} {'Within ±2':>10} {'In range':>9} "
          f"{'Width':>6} {'Avg spread':>11} {'Round (≥20)':>12} {'Failed':>7} {'Tokens/frame':>12}")
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
        spans   = [(lo, hi, f["truth"]) for i, f in enumerate(frames)
                   if f.get("truth") is not None and f["truth"] >= 0
                   for lo, hi in ranges[(i, a)] if lo >= 0]
        inr_s   = f"{100 * sum(lo <= t <= hi for lo, hi, t in spans) / len(spans):.0f}%" if spans else "—"
        width_s = f"{mean(hi - lo for lo, hi, _ in spans):.1f}" if spans else "—"
        print(f"{a:<12} {err_s:>10} {bias_s:>6} {close_s:>10} {inr_s:>9} {width_s:>6} "
              f"{mean(spreads) if spreads else 0:>11.1f} {round_:>12} {failed:>7} "
              f"{mean(tokens[a]):>12.0f}")
    print("═" * 108)
    print("Error and bias = midpoint count − hand count (bias < 0 means undercounting). In range =")
    print("share of calls whose sure..sure+maybe range holds the hand count; Width = average range")
    print("width. Spread = max − min across identical calls on one frame. Round = share of counts")
    print("≥20 that are multiples of 5 (≈20% if nothing is rounded). Failed = no usable count.")


if __name__ == "__main__":
    main()
