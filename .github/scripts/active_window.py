#!/usr/bin/env python3
"""
active_window.py — cheap pre-flight gate for the Surf Sample workflow.

Emits `active=true|false` in GITHUB_OUTPUT format. The workflow gates dependency
installation on it, so a tick outside the sampling window costs seconds instead
of minutes of Actions time.

Deliberately stdlib-only so it can run before `pip install -r requirements.txt`.
It approximates all ten SoCal spots from one representative location and widens
the result with a safety margin — src/scheduler.py still does the authoritative
per-spot check before anything is captured.
"""

import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

# Representative point for the SoCal spot cluster (Lower Trestles). Every
# configured spot is within ~70 miles, i.e. a few minutes of solar time.
LAT, LNG = 33.3814, -117.5897

# 1h operational buffer (matches config/settings.json) plus 30min of margin to
# absorb the single-point approximation and scheduler jitter.
BUFFER = timedelta(hours=1, minutes=30)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _julian_day(year: int, month: int, day: int) -> float:
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5


def _from_julian(j: float) -> datetime:
    return _EPOCH + timedelta(days=j - 2440587.5)


def sun_times(day: date, lat: float, lng: float):
    """Sunrise/sunset as UTC datetimes for the solar day starting on `day`."""
    # +0.5 converts the 0h-UT Julian date to the Julian day *number* (noon-based)
    # that the NOAA sunrise equation is defined against.
    n = _julian_day(day.year, day.month, day.day) + 0.5 - 2451545.0 + 0.0008
    j_star = n - lng / 360.0

    m = (357.5291 + 0.98560028 * j_star) % 360.0
    m_rad = math.radians(m)
    c = 1.9148 * math.sin(m_rad) + 0.0200 * math.sin(2 * m_rad) + 0.0003 * math.sin(3 * m_rad)
    lam = math.radians((m + c + 282.9372) % 360.0)

    j_transit = 2451545.0 + j_star + 0.0053 * math.sin(m_rad) - 0.0069 * math.sin(2 * lam)
    decl = math.asin(math.sin(lam) * math.sin(math.radians(23.4397)))

    lat_rad = math.radians(lat)
    cos_omega = (math.sin(math.radians(-0.833)) - math.sin(lat_rad) * math.sin(decl)) / (
        math.cos(lat_rad) * math.cos(decl)
    )
    if not -1.0 <= cos_omega <= 1.0:
        return None, None  # polar day/night; unreachable at these latitudes

    omega = math.degrees(math.acos(cos_omega)) / 360.0
    return _from_julian(j_transit - omega), _from_julian(j_transit + omega)


def is_active(now: datetime) -> bool:
    # Also test yesterday's solar day: just after UTC midnight we are still
    # inside the previous local day's window, since Pacific is UTC-7/-8.
    for offset in (-1, 0):
        rise, sset = sun_times((now + timedelta(days=offset)).date(), LAT, LNG)
        if rise and sset and rise - BUFFER <= now <= sset + BUFFER:
            return True
    return False


def main() -> None:
    now = datetime.now(timezone.utc)
    forced = os.environ.get("FORCE_RUN", "").lower() == "true"
    active = forced or is_active(now)

    rise, sset = sun_times(now.date(), LAT, LNG)
    print(f"UTC now:  {now:%Y-%m-%d %H:%M:%S}", file=sys.stderr)
    if rise and sset:
        print(f"Window:   {rise - BUFFER:%m-%d %H:%M} .. {sset + BUFFER:%m-%d %H:%M} UTC", file=sys.stderr)
    print(f"Active:   {active}{' (forced)' if forced else ''}", file=sys.stderr)

    print(f"active={'true' if active else 'false'}")


if __name__ == "__main__":
    main()
