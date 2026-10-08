#!/usr/bin/env python3
"""
verify_stack.py — health check for the collector's Supabase dependencies.

Run after enabling RLS, restoring a suspended project, or any time the
dashboard looks wrong:

    python3 scripts/verify_stack.py

Uses only the public anon key from docs/index.html, so it needs no secrets and
is safe to run from anywhere. Checks, in order:

  1. the project answers at all (a restricted project returns HTTP 402)
  2. anon can SELECT            -- the dashboard depends on this
  3. anon CANNOT INSERT         -- i.e. RLS is enabled (migration 004)
  4. how stale the newest observation is
  5. every enabled spot is keeping up -- one dead cam cannot hide behind
     nine healthy ones

Exits non-zero if any check fails, so it can be wired into CI later.
"""

import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parent.parent / "docs" / "index.html"
SPOTS = Path(__file__).resolve().parent.parent / "docs" / "spots.json"
STALE_HOURS = 3  # a healthy 15-min schedule should never exceed this by much


def load_creds() -> tuple[str, str]:
    html = DASHBOARD.read_text()
    url = re.search(r"https://[a-z0-9]+\.supabase\.co", html)
    key = re.search(r"eyJ[A-Za-z0-9_.-]+", html)
    if not (url and key):
        sys.exit(f"could not find Supabase URL/anon key in {DASHBOARD}")
    return url.group(0), key.group(0)


def request(url: str, key: str, path: str, method: str = "GET", body: dict | None = None):
    """Return (status, parsed_body). Never raises on HTTP error status."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{url}{path}", data=data, method=method)
    req.add_header("apikey", key)
    req.add_header("Authorization", f"Bearer {key}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"null")
        except json.JSONDecodeError:
            return e.code, {"raw": raw.decode(errors="replace")[:200]}
    except Exception as e:  # network / DNS / TLS
        return 0, {"message": str(e)}


def main() -> None:
    url, key = load_creds()
    print(f"project: {url}\n")
    failures = []

    def report(ok: bool, label: str, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    # 1 + 2. Reachability and anon read.
    status, body = request(url, key, "/rest/v1/observations?select=captured_at"
                                     "&order=captured_at.desc&limit=1")
    if status == 402:
        report(False, "project active", f"restricted -- {body.get('message', body)}")
        print("\nProject is suspended; the remaining checks cannot run.")
        sys.exit(1)
    report(status == 200, "anon can read", f"HTTP {status}")

    # 3. Anon must NOT be able to write. The empty payload cannot create a row:
    #    with RLS off it is stopped by the not-null constraint, with RLS on it
    #    is refused outright.
    status, body = request(url, key, "/rest/v1/observations", "POST", {})
    code = body.get("code") if isinstance(body, dict) else None
    blocked = status in (401, 403) or code == "42501"
    detail = (
        f"HTTP {status} -- writes refused"
        if blocked
        else f"HTTP {status} code={code} -- ANON CAN WRITE, apply db/migrations/004_enable_rls.sql"
    )
    report(blocked, "anon cannot write", detail)

    # 4. Freshness.
    status, body = request(url, key, "/rest/v1/observations?select=captured_at"
                                     "&order=captured_at.desc&limit=1")
    if status == 200 and body:
        newest = datetime.fromisoformat(body[0]["captured_at"])
        age_h = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
        report(
            age_h <= STALE_HOURS,
            "data is fresh",
            f"newest observation {newest:%Y-%m-%d %H:%M} UTC ({age_h:.1f}h ago)",
        )
    else:
        newest = None
        report(False, "data is fresh", f"could not read newest row (HTTP {status})")

    # 5. Per-spot freshness. Measured against the newest row overall rather than
    #    the wall clock, so the overnight gap (when every spot is idle) is not
    #    reported as N separate failures.
    if newest:
        spots = [s["id"] for s in json.loads(SPOTS.read_text())["spots"] if s.get("enabled")]
        lagging = []
        for spot_id in spots:
            status, body = request(url, key, f"/rest/v1/observations?select=captured_at"
                                             f"&spot_id=eq.{spot_id}&order=captured_at.desc&limit=1")
            if status != 200:
                lagging.append(f"{spot_id} (HTTP {status})")
            elif not body:
                lagging.append(f"{spot_id} (no rows)")
            else:
                behind_h = (newest - datetime.fromisoformat(body[0]["captured_at"])).total_seconds() / 3600
                if behind_h > STALE_HOURS:
                    lagging.append(f"{spot_id} ({behind_h:.0f}h behind)")
        report(
            not lagging,
            "every spot reporting",
            f"all {len(spots)} spots current" if not lagging else "lagging: " + ", ".join(lagging),
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
