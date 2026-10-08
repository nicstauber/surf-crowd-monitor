-- Migration 005: Surfline regional written forecasts
--
-- Surfline forecasters publish a written report per subregion (e.g. "North
-- Orange County Forecast"), typically updated morning and afternoon. Every
-- spot in a subregion shares the same report, so it is stored once per
-- (subregion, published_at) rather than copied onto each observation.
-- Join to observations via the spot's subregion_id in docs/spots.json and the
-- latest published_at <= captured_at.

CREATE TABLE IF NOT EXISTS regional_reports (
    subregion_id   text        NOT NULL,
    published_at   timestamptz NOT NULL,
    subregion_name text,
    forecaster     text,
    headline       text,
    body_html      text,
    note_html      text,
    day_to_watch   boolean,
    fetched_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (subregion_id, published_at)
);

-- Same posture as observations (migration 004): anon may read, only the
-- service_role collector may write.
ALTER TABLE regional_reports ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "anon read regional_reports" ON regional_reports;
CREATE POLICY "anon read regional_reports"
    ON regional_reports
    FOR SELECT
    TO anon
    USING (true);
