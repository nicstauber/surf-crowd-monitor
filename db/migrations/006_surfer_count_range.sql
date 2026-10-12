-- Migration 006: surfer count as a range
--
-- Far-off surfers are a few pixels that could also be birds, buoys or wave
-- shadows, so the counter reports surfers it is sure of and possible ones.
--   surfer_count_min = sure surfers
--   surfer_count_max = sure + possible surfers
--   surfer_count     = midpoint, kept for stats, heatmaps and alert thresholds
-- Rows before this migration have NULL min/max; their surfer_count is a single
-- estimate (and, before count_method ..._v8_surfers_only, included waders).

ALTER TABLE observations
  ADD COLUMN IF NOT EXISTS surfer_count_min INTEGER,
  ADD COLUMN IF NOT EXISTS surfer_count_max INTEGER;
