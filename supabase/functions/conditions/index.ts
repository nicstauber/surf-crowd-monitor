/**
 * conditions — Supabase Edge Function
 *
 * Proxies the Surfline forecast API server-side, avoiding Cloudflare's
 * CORS preflight block on browser requests and IP block on CI runners.
 *
 * GET /functions/v1/conditions?spotId=<surfline_spot_id>
 *
 * Returns JSON with parsed conditions ready for the dashboard and DB writer.
 * Requires the Supabase anon key in the Authorization or apikey header
 * (standard Supabase client behaviour — no extra config needed).
 */

const SURFLINE_BASE = "https://services.surfline.com/kbyg/spots/forecasts";
const SURFLINE_REPORTS = "https://services.surfline.com/kbyg/spots/reports";

// Browser-like headers — keeps Cloudflare happy on server-side requests
const SURFLINE_HEADERS = {
  "origin": "https://www.surfline.com",
  "referer": "https://www.surfline.com/",
  "user-agent":
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " +
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
  "accept": "application/json, text/plain, */*",
  "accept-language": "en-US,en;q=0.9",
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
};

function closest(arr: Record<string, unknown>[], ts: number) {
  if (!arr?.length) return null;
  return arr.reduce((a, b) =>
    Math.abs((b.timestamp as number) - ts) <
      Math.abs((a.timestamp as number) - ts)
      ? b
      : a
  );
}

// Why each upstream call failed on its last attempt (HTTP status or error
// message), keyed by endpoint. Returned to the caller and stored in
// conditions_raw so intermittent gaps can be diagnosed from the database.
type Failures = Record<string, string>;

// Surfline intermittently fails single requests (a different endpoint each
// time), so retry a few times before giving up on a field.
async function getJson(
  name: string,
  url: string,
  failures: Failures,
): Promise<Record<string, unknown> | null> {
  const delaysMs = [0, 400, 1200];
  for (const delay of delaysMs) {
    if (delay) await new Promise((res) => setTimeout(res, delay));
    try {
      const r = await fetch(url, { headers: SURFLINE_HEADERS });
      if (r.ok) {
        delete failures[name];
        return await r.json();
      }
      failures[name] = `HTTP ${r.status}`;
      await r.body?.cancel();
    } catch (e) {
      failures[name] = String(e).slice(0, 120);
    }
  }
  return null;
}

Deno.serve(async (req) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: CORS_HEADERS });
  }

  const { searchParams } = new URL(req.url);

  const spotId = searchParams.get("spotId");

  if (!spotId) {
    return new Response(JSON.stringify({ error: "spotId is required" }), {
      status: 400,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    });
  }

  const params = `spotId=${spotId}&days=1&intervalHours=1`;
  const ts = Date.now() / 1000;

  // /wave was retired (404) around 2026-09-01; surf height and swells are
  // now separate endpoints.
  const failures: Failures = {};
  const [surfRes, swellsRes, windRes, tidesRes, ratingRes, reportRes] = await Promise.all([
    getJson("surf",   `${SURFLINE_BASE}/surf?${params}`,     failures),
    getJson("swells", `${SURFLINE_BASE}/swells?${params}`,   failures),
    getJson("wind",   `${SURFLINE_BASE}/wind?${params}`,     failures),
    getJson("tides",  `${SURFLINE_BASE}/tides?${params}`,    failures),
    getJson("rating", `${SURFLINE_BASE}/rating?${params}`,   failures),
    getJson("report", `${SURFLINE_REPORTS}?spotId=${spotId}`, failures),
  ]);

  // deno-lint-ignore no-explicit-any
  const data = (res: any, key: string) => res?.data?.[key] ?? [];

  const surf   = closest(data(surfRes,   "surf"),   ts) as any;
  const swells = closest(data(swellsRes, "swells"), ts) as any;
  const wind   = closest(data(windRes,   "wind"),   ts) as any;
  const tide   = closest(data(tidesRes,  "tides"),  ts) as any;
  const rating = closest(data(ratingRes, "rating"), ts) as any;

  // Swell slots are no longer ordered by importance (slot 0 is often an empty
  // placeholder), so take the one with the highest impact.
  // deno-lint-ignore no-explicit-any
  const swell = (swells?.swells ?? []).filter((s: any) => s.height > 0)
    // deno-lint-ignore no-explicit-any
    .reduce((a: any, b: any) => (a && a.impact >= b.impact ? a : b), null);

  // Regional written forecast (e.g. "North Orange County Forecast"). Shared by
  // every spot in the subregion; updated by Surfline's forecasters ~AM and PM.
  // deno-lint-ignore no-explicit-any
  const rep = (reportRes as any)?.report;
  // deno-lint-ignore no-explicit-any
  const subregionUrl: string = (reportRes as any)?.associated?.subregionUrl ?? "";
  const [subregionSlug, subregionId] = subregionUrl.split("/").slice(-2);
  const report = rep?.timestamp && subregionId
    ? {
      subregion_id:   subregionId,
      subregion_name: subregionSlug,
      published_at:   new Date(rep.timestamp * 1000).toISOString(),
      forecaster:     rep.forecaster?.name ?? null,
      headline:       rep.headline ?? null,
      body_html:      rep.body ?? null,
      // deno-lint-ignore no-explicit-any
      note_html:      (reportRes as any)?.notes?.subregion ?? null,
      day_to_watch:   rep.dayToWatch ?? null,
    }
    : null;

  const result = {
    // Surf / wave
    wave_height_min:     surf?.surf?.min              ?? null,
    wave_height_max:     surf?.surf?.max              ?? null,
    surf_human_relation: surf?.surf?.humanRelation    ?? null,
    // Swell (dominant = highest impact)
    swell_height:        swell?.height                ?? null,
    swell_period:        swell?.period                ?? null,
    swell_direction:     swell?.direction             ?? null,
    // Wind
    wind_speed:          wind?.speed                  ?? null,
    wind_direction:      wind?.direction              ?? null,
    wind_direction_type: wind?.directionType          ?? null,
    // Tide
    tide_height:         tide?.height                 ?? null,
    // Rating
    spot_rating:         rating?.rating?.key          ?? null,
    // Regional written report (null if unavailable)
    report,
    // Endpoints that still failed after retries, e.g. {"wind": "HTTP 429"}
    upstream_failures: failures,
  };

  const complete = Object.keys(failures).length === 0;
  return new Response(JSON.stringify(result), {
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json",
      // Cache complete results 5 min at the CDN edge; never cache a partial
      // one, or every caller for the next 5 min inherits the gap.
      "Cache-Control": complete ? "public, max-age=300" : "no-store",
    },
  });
});
