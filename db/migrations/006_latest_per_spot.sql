-- Migration 006: latest observation for each requested spot, in one call
--
-- The dashboard used to send one "newest row for this spot" request per spot
-- on every load and every 5-minute refresh (19 requests per open tab per
-- refresh). This returns the same rows in a single request.
--
-- Each spot is one index lookup on idx_observations_spot_captured. Like
-- heatmap_daily_avg() it runs as the calling role, so the anon SELECT policy
-- from migration 004 applies. Callers trim columns with PostgREST's select=.

CREATE OR REPLACE FUNCTION latest_per_spot(p_spot_ids TEXT[])
RETURNS SETOF observations
LANGUAGE sql STABLE
SET search_path = ''
AS $$
  SELECT o.*
  FROM unnest(p_spot_ids) AS s(spot_id)
  CROSS JOIN LATERAL (
    SELECT *
    FROM public.observations
    WHERE observations.spot_id = s.spot_id
    ORDER BY captured_at DESC
    LIMIT 1
  ) o;
$$;
