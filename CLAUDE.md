# Surf Crowd Monitor

## Goal
Capture frames from Surfline surf cams on a scheduled basis, count surfers in the water using Claude vision API, and store results in a cloud database for later analysis and alerting.

## Current State
Working spike script: `surf_spike_v6.py`

### What it does
1. Fetches a 3-frame burst from a Surfline HLS stream (no auth required — just a `referer` header)
2. Scores each frame for quality using Laplacian variance + patch noise (glare detection)
3. Sends each qualifying frame to Claude vision API with a prompt that counts people in the surf zone beyond the shore break
4. Returns the max count across the 3 frames plus per-frame detail
5. Outputs a contact sheet JPEG to `spike_output/` for visual verification

### Key decisions already made
- **Detection:** Claude vision on zoomed tiles — the frame is cut into a grid (`tile_grid` in `config/settings.json`, default 3x2), each tile upscaled to `tile_width` and sent separately; no manual crop zones. Conditions come from one separate full-frame call
- **Count = number of points located across tiles, never a number Claude states.** Whole-frame counting guessed: Haiku 4.5 snapped crowds to favourite values (731 rows of exactly 47; 12 and 28 overrepresented — treat pre-Oct-8-2026 counts as coarse), Haiku 5.5 rounded big crowds to multiples of 5, and whole-frame point lists put dots on empty water. Score any detection change with `scripts/eval_counts.py` against the hand counts in `eval/frames.json` (from the Lineup Answer Key artifact); it runs in CI as "Count Eval"
- **Count method:** Max across 3-frame burst (not average)
- **Quality scoring:** Laplacian variance + patch noise (glare-aware); frames below threshold excluded from API calls
- **Model:** `claude-haiku-5-5` at effort `medium` (Oct 2026 A/B vs 4.5 on hand-counted frames: ~2x smaller count error, ~5x cheaper; `scripts/compare_models.py`)
- **Database:** Postgres on Supabase
- **Cost controls:** conditions assessed once an hour per spot (count-only calls in between, conditions carried forward, `count_method` ends in `_count_only`); lineups with ≤2 surfers sampled every 30 min instead of 15. Tunable in `config/settings.json`
- **Language:** Python

## Surf Spots
Listed north to south — the same order as `docs/spots.json`, which sets the dashboard order.

| Key | Display Name | Surfline slug |
|-----|-------------|--------------|
| `malibu` | Malibu - Surfrider Beach | `wc-malibusurfrider` |
| `el_porto` | El Porto - 42nd Street | `wc-elporto42nd` |
| `seal_beach_jetty` | Seal Beach Jetty | `wc-sealbeachjetty4k` |
| `bolsa_chica_tower17` | Bolsa Chica - Tower 17 | `wc-tower17southbolsa` |
| `hb_cliffs` | North HB - Goldenwest | `wc-goldenwest` |
| `north_hb_20th` | North HB - 20th Street | `wc-twentiethst` |
| `hb_pier_south_tower1` | HB Pier South - Tower 1-3 | `wc-hbpierss` |
| `hb_pier_south` | HB Pier South | `wc-hbpierssov` |
| `river_jetties_north` | River Jetties - North | `wc-riverjettiesnorth` |
| `newport_56th` | Newport 56th | `wc-fiftysixnewport` |
| `newport_54th` | Newport 54th | `wc-fiftyfournewport` |
| `newport_36th` | Newport 36th | `wc-36thstnewport` |
| `newport_point` | Newport Point | `wc-newportpierss` |
| `the_wedge` | The Wedge | `wc-wedgeov` |
| `corona_del_mar` | Corona del Mar | `wc-coronadelmar` |
| `salt_creek` | Salt Creek | `wc-saltcreek` |
| `doheny_second_spot` | Doheny - The Hammer | `wc-secondspotdoheny` |
| `upper_trestles` | Upper Trestles | `wc-upperstrestles` |
| `lower_trestles` | Lower Trestles | `wc-lowers` |

All HLS streams follow: `https://hls.cdn-surfline.com/oregon/[slug]/playlist.m3u8`

## Environment
- `ANTHROPIC_API_KEY` is set in the environment — do not prompt the user to set it
- Running on macOS (development); target deployment TBD

## Next Steps (not yet built)
- [ ] Proper project structure — separate modules for config, scheduler, DB writer, vision
- [ ] Supabase integration — write surfer counts + frame metadata to Postgres
- [ ] Sunrise/sunset-aware scheduler — sample every 15 min from 1hr before sunrise to 1hr after sunset
- [ ] Surfline conditions enrichment — fetch wave height, wind, and tide via the unofficial Surfline API per sample
- [ ] Multi-spot orchestration — run all configured spots per scheduled tick

## Surfline API Notes
- HLS streams require `origin: https://www.surfline.com` and `referer: https://www.surfline.com/` headers
- No authentication needed for the camera streams
- Conditions come from the unofficial Surfline REST API via the `conditions` Supabase Edge Function (Surfline's Cloudflare blocks CI/sandbox IPs; Supabase is not blocked). Endpoints confirmed working Oct 2026, all under `https://services.surfline.com/kbyg/spots/`:
  - **The function makes one call per spot: `reports?spotId=`.** Its `forecast` holds current `waveHeight`, `swells` (unordered — pick by max `power`), `wind`, `tide.current`, `conditions.value` (rating), `waterTemp`; its `report` holds the regional written forecast (`headline`, `body` HTML, `forecaster`, `timestamp`; subregion ID = last segment of `associated.subregionUrl`), stored in the `regional_reports` table.
  - Don't fan out to several endpoints per spot: six parallel calls (`forecasts/surf`, `/swells`, `/wind`, `/tides`, `/rating` + `reports`) drew Cloudflare HTTP 403s on random calls, even with retries (Oct 2026).
  - `forecasts/wave` was retired (404) around 2026-09-01
