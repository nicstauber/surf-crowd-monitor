-- Migration 003: server-side daily average aggregation for the crowd heatmap
--
-- Replaces client-side aggregation in docs/index.html that was hitting Supabase's
-- 1000-row default PostgREST limit and silently dropping the most recent days.
--
-- Returns one row per (spot_id, local date) — at most 10 spots × 30 days = 300 rows.
-- Dates are in America/Los_Angeles local time to match the browser's localDateStr().

CREATE OR REPLACE FUNCTION heatmap_daily_avg(p_days INT DEFAULT 30)
RETURNS TABLE (
  spot_id   TEXT,
  date_str  TEXT,
  avg_count INT
)
LANGUAGE sql STABLE
AS $$
  SELECT
    spot_id,
    TO_CHAR(
      DATE_TRUNC('day', captured_at AT TIME ZONE 'America/Los_Angeles'),
      'YYYY-MM-DD'
    ) AS date_str,
    ROUND(AVG(surfer_count))::INT AS avg_count
  FROM observations
  WHERE
    captured_at >= NOW() - (p_days || ' days')::INTERVAL
    AND count_reliable = TRUE
    AND surfer_count IS NOT NULL
  GROUP BY spot_id, date_str
  ORDER BY spot_id, date_str;
$$;
