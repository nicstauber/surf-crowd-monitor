-- Migration 004: enable row level security on observations
--
-- The dashboard talks to PostgREST with the anon key, which is embedded in
-- docs/index.html and is therefore served to every visitor of the site. With
-- RLS disabled that key grants anon full INSERT / UPDATE / DELETE on this
-- table -- a single request could drop the entire observation history.
--
-- The collector writes with SUPABASE_SERVICE_KEY (the service_role key), which
-- bypasses RLS entirely, so the pipeline is unaffected by these policies.
--
-- heatmap_daily_avg() is a plain STABLE sql function with no SECURITY DEFINER,
-- so it executes as the calling role and is covered by the SELECT policy below.

ALTER TABLE observations ENABLE ROW LEVEL SECURITY;

-- The dashboard only ever reads.
DROP POLICY IF EXISTS "anon read observations" ON observations;
CREATE POLICY "anon read observations"
    ON observations
    FOR SELECT
    TO anon
    USING (true);

-- Deliberately no INSERT / UPDATE / DELETE policy: with RLS enabled and no
-- matching policy, PostgreSQL denies those operations for anon by default.
