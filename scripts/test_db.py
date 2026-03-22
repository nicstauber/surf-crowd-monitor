"""
Quick Supabase connectivity test.

Usage:
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python scripts/test_db.py

Or with a .env file:
    python scripts/test_db.py
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import db


def main():
    print("1. Connecting to Supabase...")
    try:
        client = db.get_client()
        print(f"   OK — connected to {os.environ['SUPABASE_URL']}")
    except EnvironmentError as e:
        print(f"   FAIL — {e}")
        sys.exit(1)

    print("2. Writing test observation...")
    record = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "spot_id": "test",
        "spot_name": "Test Spot",
        "surfer_count": 0,
        "count_reliable": False,
        "count_method": "test",
        "claude_notes": "connectivity test — safe to delete",
        "session_quality": "test",
    }
    row_id = db.write_observation(record)
    if row_id:
        print(f"   OK — inserted row id={row_id}")
    else:
        print("   FAIL — insert returned None (check logs above)")
        sys.exit(1)

    print("3. Reading back test observations...")
    rows = db.latest_observations("test", limit=3)
    if rows:
        print(f"   OK — got {len(rows)} row(s)")
        for r in rows:
            print(f"        {r['id']} | {r['captured_at']} | {r['claude_notes']}")
    else:
        print("   FAIL — no rows returned")
        sys.exit(1)

    print("\nAll checks passed. You can delete the test rows from Supabase.")


if __name__ == "__main__":
    main()
