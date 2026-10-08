"""
Side-by-side surfer counts from two Claude models on the exact same frames.

Captures a fresh full-resolution burst from each enabled spot, picks the best
frame (same quality scoring as the scheduler), and sends that one frame to
every model "arm" below. Prints how often the counts agree and what each arm
costs per call, using the token counts the API actually reports.

Frames are saved to spike_output/model_compare/<timestamp>/ so you can eyeball
the disagreements and re-run the comparison later without recapturing.

Usage:
    python scripts/compare_models.py                    # 1 round, all enabled spots
    python scripts/compare_models.py --rounds 3         # ~30 frames
    python scripts/compare_models.py --spots malibu,el_porto
    python scripts/compare_models.py --frames-dir spike_output/model_compare/2026-10-08T15-00-00

Needs ANTHROPIC_API_KEY (env or .env). No Supabase access needed — nothing is
written to the database.
"""

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic
from PIL import Image

from capture import fetch_burst, score_frames
from detect  import analyze_frame

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

_ROOT = Path(__file__).resolve().parent.parent

# (label, model, effort). The first arm is the baseline the others are compared to.
ARMS = [
    ("haiku-4.5",        "claude-haiku-4-5", None),
    ("haiku-5.5 low",    "claude-haiku-5-5", "low"),
    ("haiku-5.5 medium", "claude-haiku-5-5", "medium"),
]

# USD per million tokens (input, output) — prompts here are far under 100K tokens.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Average 15-min ticks per day across the year (sunrise-1h .. sunset+1h in SoCal).
TICKS_PER_DAY = 57


def call_cost(model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = PRICES[model]
    return (in_tok * p_in + out_tok * p_out) / 1_000_000


def capture_best_frame(spot: dict, settings: dict, tmp_dir: Path, out_path: Path):
    """Fetch a burst and save the best qualifying frame at full resolution."""
    paths = fetch_burst(spot, settings, tmp_dir / spot["id"])
    if not paths:
        return None
    qualifying = [f for f in score_frames(paths, settings) if f["quality"]["contributes"]]
    if not qualifying:
        return None
    best = max(qualifying, key=lambda f: f["quality"]["overall_score"])
    best["image"].save(out_path, format="JPEG", quality=95)
    return out_path


def capture_frames(spots, settings, run_dir: Path, rounds: int) -> list[Path]:
    tmp_dir = run_dir / "_bursts"
    saved   = []
    for r in range(1, rounds + 1):
        print(f"📸 Round {r}/{rounds}: capturing {len(spots)} spots in parallel...")
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                s["id"]: pool.submit(capture_best_frame, s, settings, tmp_dir,
                                     run_dir / f"{s['id']}__r{r}.jpg")
                for s in spots
            }
        for spot_id, fut in futures.items():
            path = fut.result()
            if path:
                saved.append(path)
            else:
                print(f"   ⚠️  {spot_id}: no usable frame (offline, dark, or glare)")
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=1, help="capture rounds (default 1)")
    ap.add_argument("--spots", help="comma-separated spot ids (default: all enabled)")
    ap.add_argument("--frames-dir", type=Path, help="reuse frames saved by an earlier run")
    ap.add_argument("--project-spots", type=int, default=22,
                    help="spot count for the monthly cost projection (default 22)")
    args = ap.parse_args()

    settings = json.loads((_ROOT / "config" / "settings.json").read_text())
    spots    = json.loads((_ROOT / "config" / "spots.json").read_text())["spots"]
    spots    = [s for s in spots if s.get("enabled")]
    if args.spots:
        wanted = set(args.spots.split(","))
        spots  = [s for s in spots if s["id"] in wanted]

    if args.frames_dir:
        run_dir = args.frames_dir
        frames  = sorted(run_dir.glob("*__r*.jpg"))
    else:
        run_dir = _ROOT / "spike_output" / "model_compare" / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        frames  = capture_frames(spots, settings, run_dir, args.rounds)

    if not frames:
        sys.exit("No frames to compare.")

    client = anthropic.Anthropic()
    rows   = []
    print(f"\n🤖 Sending {len(frames)} frames × {len(ARMS)} models...\n")

    for path in frames:
        img = Image.open(path).convert("RGB")
        row = {"frame": path.name, "spot": path.name.split("__")[0]}
        for label, model, effort in ARMS:
            arm_settings = {**settings, "claude_model": model, "claude_effort": effort}
            res = analyze_frame(img, f"{row['spot']}/{label}", client, arm_settings)
            in_tok, out_tok = res.get("input_tokens", 0), res.get("output_tokens", 0)
            row[label] = {
                "count":   res["count"],
                "in_tok":  in_tok,
                "out_tok": out_tok,
                "cost":    call_cost(model, in_tok, out_tok) if in_tok else 0.0,
            }
        rows.append(row)
        counts = "  ".join(f"{label}={row[label]['count']:>3}" for label, _, _ in ARMS)
        print(f"  {row['spot']:<22} {counts}")

    # ── Summary ───────────────────────────────────────────────────────────────
    base = ARMS[0][0]
    calls_per_month = args.project_spots * TICKS_PER_DAY * 30
    print("\n" + "═" * 78)
    print(f"{'Model':<18} {'Exact':>6} {'±1':>6} {'±2':>6} {'Avg diff':>9} "
          f"{'In tok':>7} {'Out tok':>8} {'$/call':>9} {f'$/mo @{args.project_spots}':>10}")
    print("─" * 78)
    for label, model, _ in ARMS:
        valid = [r for r in rows if r[label]["count"] >= 0 and r[base]["count"] >= 0]
        diffs = [abs(r[label]["count"] - r[base]["count"]) for r in valid]
        ok    = [r[label] for r in rows if r[label]["in_tok"]]
        if not ok:
            print(f"{label:<18} all calls failed")
            continue
        cost = mean(a["cost"] for a in ok)
        pct  = lambda n: f"{100 * sum(d <= n for d in diffs) / len(diffs):.0f}%" if diffs else "—"
        print(f"{label:<18} {pct(0):>6} {pct(1):>6} {pct(2):>6} "
              f"{(mean(diffs) if diffs else 0):>9.2f} "
              f"{mean(a['in_tok'] for a in ok):>7.0f} {mean(a['out_tok'] for a in ok):>8.0f} "
              f"{cost:>9.5f} {cost * calls_per_month:>10.2f}")
    print("═" * 78)
    print(f"Agreement is measured against {base}, which is not ground truth —")
    print("open the biggest disagreements below and count them yourself.\n")

    # Biggest disagreements first, so the frames worth eyeballing are on top.
    def spread(r):
        c = [r[label]["count"] for label, _, _ in ARMS if r[label]["count"] >= 0]
        return max(c) - min(c) if c else 0
    for r in sorted(rows, key=spread, reverse=True)[:5]:
        if spread(r) == 0:
            break
        counts = ", ".join(f"{label}={r[label]['count']}" for label, _, _ in ARMS)
        print(f"  🔍 {run_dir / r['frame']}\n     {counts}")

    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "spot"] + [f"{label} {k}" for label, _, _ in ARMS
                                        for k in ("count", "in_tok", "out_tok", "cost")])
        for r in rows:
            w.writerow([r["frame"], r["spot"]] + [r[label][k] for label, _, _ in ARMS
                                                  for k in ("count", "in_tok", "out_tok", "cost")])
    print(f"\n📄 Full results: {csv_path}")


if __name__ == "__main__":
    main()
