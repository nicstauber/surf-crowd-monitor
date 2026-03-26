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
- **Detection:** Claude vision, full-frame — no manual crop zones or tile splitting
- **Count method:** Max across 3-frame burst (not average)
- **Quality scoring:** Laplacian variance + patch noise (glare-aware); frames below threshold excluded from API calls
- **Model:** `claude-haiku-4-5`
- **Database:** Postgres on Supabase
- **Language:** Python

## Surf Spots
| Key | Surfline slug |
|-----|--------------|
| `lower_trestles` | `wc-lowerslefts` |
| `upper_trestles` | `wc-upperstrestles` |
| `doheny_second_spot` | `wc-secondspotdoheny` |
| `hb_pier_south` | `wc-huntingtonbeachsouthside` |
| `hb_cliffs` | `wc-huntingtoncliffs` |
| `newport_56th` | `wc-fiftysixnewport` |
| `el_porto` | `wc-elporto42nd` |
| `malibu` | `wc-malibusurfrider` |

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
- Conditions data (wave height, wind, tide) is available via the unofficial Surfline REST API — endpoints to be confirmed when building the enrichment step
