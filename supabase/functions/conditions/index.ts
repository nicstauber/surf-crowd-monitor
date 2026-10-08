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

// One call per spot. The spot report carries current wave height, swells,
// wind, tide and rating plus the regional written forecast. Fetching the
// separate /forecasts/{surf,swells,wind,tides,rating} endpoints in parallel
// (six calls per spot) tripped Cloudflare: random calls came back HTTP 403
// even after retries. (/forecasts/wave was retired, 404, around 2026-09-01.)
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

// Retry with backoff; returns the parsed body, or null plus the reason the
// last attempt failed (HTTP status or error message).
async function getJson(
  url: string,
): Promise<{ body: Record<string, unknown> | null; failure: string | null }> {
  let failure: string | null = null;
  for (const delay of [0, 1500, 4000]) {
    if (delay) await new Promise((res) => setTimeout(res, delay));
    try {
      const r = await fetch(url, { headers: SURFLINE_HEADERS });
      if (r.ok) return { body: await r.json(), failure: null };
      failure = `HTTP ${r.status}`;
      await r.body?.cancel();
    } catch (e) {
      failure = String(e).slice(0, 120);
    }
  }
  return { body: null, failure };
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

  const { body, failure } = await getJson(`${SURFLINE_REPORTS}?spotId=${spotId}`);

  // deno-lint-ignore no-explicit-any
  const res = body as any;
  const fc = res?.forecast;

  // Swell slots are unordered; the dominant one has the most power.
  // deno-lint-ignore no-explicit-any
  const swell = (fc?.swells ?? []).filter((s: any) => s.height > 0)
    // deno-lint-ignore no-explicit-any
    .reduce((a: any, b: any) => (a && a.power >= b.power ? a : b), null);

  // Regional written forecast (e.g. "North Orange County Forecast"). Shared by
  // every spot in the subregion; updated by Surfline's forecasters ~AM and PM.
  const rep = res?.report;
  const subregionUrl: string = res?.associated?.subregionUrl ?? "";
  const [subregionSlug, subregionId] = subregionUrl.split("/").slice(-2);
  const report = rep?.timestamp && subregionId
    ? {
      subregion_id:   subregionId,
      subregion_name: subregionSlug,
      published_at:   new Date(rep.timestamp * 1000).toISOString(),
      forecaster:     rep.forecaster?.name ?? null,
      headline:       rep.headline ?? null,
      body_html:      rep.body ?? null,
      note_html:      res?.notes?.subregion ?? null,
      day_to_watch:   rep.dayToWatch ?? null,
    }
    : null;

  const result = {
    // Surf / wave (current; camera- or forecaster-observed when available)
    wave_height_min:     fc?.waveHeight?.min           ?? null,
    wave_height_max:     fc?.waveHeight?.max           ?? null,
    surf_human_relation: fc?.waveHeight?.humanRelation ?? null,
    wave_height_source:  fc?.waveHeight?.type          ?? null,
    // Swell (dominant = highest power)
    swell_height:        swell?.height                 ?? null,
    swell_period:        swell?.period                 ?? null,
    swell_direction:     swell?.direction              ?? null,
    // Wind
    wind_speed:          fc?.wind?.speed               ?? null,
    wind_gust:           fc?.wind?.gust                ?? null,
    wind_direction:      fc?.wind?.direction           ?? null,
    wind_direction_type: fc?.wind?.directionType       ?? null,
    // Tide
    tide_height:         fc?.tide?.current?.height     ?? null,
    // Rating
    spot_rating:         fc?.conditions?.value         ?? null,
    // Water / weather
    water_temp_f:        fc?.waterTemp?.max            ?? null,
    air_temp_f:          fc?.weather?.temperature      ?? null,
    // Regional written report (null if unavailable)
    report,
    // Why the Surfline call failed after retries, e.g. {"report": "HTTP 403"}
    upstream_failures:   failure ? { report: failure } : {},
  };

  return new Response(JSON.stringify(result), {
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json",
      // Cache complete results 5 min at the CDN edge; never cache a failure,
      // or every caller for the next 5 min inherits the gap.
      "Cache-Control": failure ? "no-store" : "public, max-age=300",
    },
  });
});
