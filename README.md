# Surf Crowd Monitor

Captures frames from Surfline surf cams, counts surfers using Claude vision, and stores results in Supabase for analysis and alerting.

## Project structure

```
config/
  spots.json              — spot definitions (edit to add/disable spots)
  settings.json           — global settings (intervals, burst config, thresholds)
src/
  scheduler.py            — main entry point, sunrise/sunset-aware loop
  capture.py              — HLS frame fetch + quality scoring
  detect.py               — Claude vision surfer counting
  conditions.py           — Surfline conditions enrichment
  db.py                   — Supabase read/write
db/
  migrations/
    001_initial.sql       — Supabase schema (run once)
spike_output/             — frame images and contact sheets written here
surf_spike_v6.py          — original spike script (reference, do not delete)
```

## Setup

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

Requires `ffmpeg` on your PATH:
```bash
brew install ffmpeg
```

### 2. Set environment variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export SUPABASE_URL=https://xxxx.supabase.co
export SUPABASE_SERVICE_KEY=eyJ...
```

`ANTHROPIC_API_KEY` is already set in the environment. Add `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` once your Supabase project is created.

### 3. Create the Supabase table

In your Supabase dashboard → SQL Editor → New query, paste and run:

```
db/migrations/001_initial.sql
```

### 4. Verify Surfline spot IDs

The `surfline_spot_id` values in `config/spots.json` need to match what Surfline uses for each spot. To find the correct ID:

1. Go to [surfline.com](https://www.surfline.com) and navigate to a spot forecast page
2. The spot ID appears in the URL: `surfline.com/surf-report/lower-trestles/<SPOT_ID>`
3. Update `config/spots.json` with the correct IDs

## Running

### Continuous mode (production)
```bash
python src/scheduler.py
```

Runs every 15 minutes during the active window (1hr before sunrise to 1hr after sunset). Sleeps outside that window.

### Single cycle (testing)
```bash
python src/scheduler.py --once
```

Runs one full sample across all enabled spots immediately and exits. If Supabase env vars are not set, the DB write is skipped and results are logged only.

## Configuration

### Adding a new spot (`config/spots.json`)

```json
{
  "id": "my_spot",
  "name": "My Spot",
  "hls_slug": "wc-myspot",
  "surfline_spot_id": "SPOT_ID_FROM_SURFLINE",
  "lat": 34.0000,
  "lng": -118.0000,
  "timezone": "America/Los_Angeles",
  "enabled": true
}
```

Set `"enabled": false` to disable a spot without removing it.

### Tuning settings (`config/settings.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `burst_frame_count` | 3 | Frames captured per sample |
| `burst_interval_seconds` | 10 | Delay between burst frames |
| `sample_interval_minutes` | 15 | How often to sample |
| `active_window_before_sunrise_hours` | 1 | Start sampling this many hours before sunrise |
| `active_window_after_sunset_hours` | 1 | Stop sampling this many hours after sunset |
| `quality_threshold` | 0.45 | Frames below this score are excluded from Claude calls |
| `claude_model` | `claude-haiku-4-5` | Model used for surfer counting |
| `image_width` | 1280 | Images are resized to this width before sending to Claude |

## Cost estimate

At default settings (3 frames × 4 spots × 64 samples/day):
- ~$1.00–$1.30/day for Claude API (Haiku pricing)
- Supabase free tier is sufficient for this write volume
