-- Surf Crowd Monitor — initial schema
-- Run this in your Supabase SQL editor (Dashboard → SQL Editor → New query)

CREATE TABLE IF NOT EXISTS observations (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    captured_at       TIMESTAMPTZ NOT NULL,
    spot_id           TEXT        NOT NULL,
    spot_name         TEXT        NOT NULL,

    -- Surfer count
    surfer_count      INTEGER,
    count_reliable    BOOLEAN,
    count_method      TEXT,

    -- Frame quality
    session_quality   FLOAT,
    frame_quality_avg FLOAT,
    lap_var_avg       FLOAT,
    noisy_pct_avg     FLOAT,

    -- Wave conditions
    wave_height_min   FLOAT,
    wave_height_max   FLOAT,
    swell_height      FLOAT,
    swell_period      INTEGER,
    swell_direction   FLOAT,

    -- Wind & tide
    wind_speed        FLOAT,
    wind_direction    FLOAT,
    tide_height       FLOAT,

    -- Surfline rating
    spot_rating       TEXT,

    -- Raw payloads
    conditions_raw    JSONB,
    frames_raw        JSONB,

    -- Claude output
    claude_notes      TEXT,

    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Primary query pattern: fetch observations for a spot ordered by time
CREATE INDEX IF NOT EXISTS idx_observations_spot_captured
    ON observations (spot_id, captured_at DESC);

-- Optional: useful for time-range queries across all spots
CREATE INDEX IF NOT EXISTS idx_observations_captured
    ON observations (captured_at DESC);
