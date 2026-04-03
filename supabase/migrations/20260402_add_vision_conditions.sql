-- Migration: add Claude vision conditions columns to observations
-- These are populated by the v7 prompt alongside the surfer count.

ALTER TABLE observations
  ADD COLUMN IF NOT EXISTS vision_surface            TEXT,
  ADD COLUMN IF NOT EXISTS vision_swell_size         TEXT,
  ADD COLUMN IF NOT EXISTS vision_wave_quality       TEXT,
  ADD COLUMN IF NOT EXISTS vision_wind_effect        TEXT,
  ADD COLUMN IF NOT EXISTS vision_crowd_distribution TEXT,
  ADD COLUMN IF NOT EXISTS vision_water_clarity      TEXT,
  ADD COLUMN IF NOT EXISTS vision_lighting           TEXT,
  ADD COLUMN IF NOT EXISTS vision_visibility         TEXT,
  ADD COLUMN IF NOT EXISTS vision_conditions_notes   TEXT;
