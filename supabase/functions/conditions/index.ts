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

async function getJson(url: string): Promise<Record<string, unknown> | null> {
  try {
    const r = await fetch(url, { headers: SURFLINE_HEADERS });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
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

  const [waveRes, windRes, tidesRes, ratingRes] = await Promise.all([
    getJson(`${SURFLINE_BASE}/wave?${params}`),
    getJson(`${SURFLINE_BASE}/wind?${params}`),
    getJson(`${SURFLINE_BASE}/tides?${params}`),
    getJson(`${SURFLINE_BASE}/rating?${params}`),
  ]);

  // deno-lint-ignore no-explicit-any
  const data = (res: any, key: string) => res?.data?.[key] ?? [];

  const wave   = closest(data(waveRes,   "wave"),   ts) as any;
  const wind   = closest(data(windRes,   "wind"),   ts) as any;
  const tide   = closest(data(tidesRes,  "tides"),  ts) as any;
  const rating = closest(data(ratingRes, "rating"), ts) as any;

  const result = {
    // Surf / wave
    wave_height_min:     wave?.surf?.min              ?? null,
    wave_height_max:     wave?.surf?.max              ?? null,
    surf_human_relation: wave?.surf?.humanRelation    ?? null,
    // Swell (dominant = index 0, sorted by impact desc)
    swell_height:        wave?.swells?.[0]?.height    ?? null,
    swell_period:        wave?.swells?.[0]?.period    ?? null,
    swell_direction:     wave?.swells?.[0]?.direction ?? null,
    // Wind
    wind_speed:          wind?.speed                  ?? null,
    wind_direction:      wind?.direction              ?? null,
    wind_direction_type: wind?.directionType          ?? null,
    // Tide
    tide_height:         tide?.height                 ?? null,
    // Rating
    spot_rating:         rating?.rating?.key          ?? null,
  };

  return new Response(JSON.stringify(result), {
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json",
      "Cache-Control": "public, max-age=300", // cache 5 min at CDN edge
    },
  });
});
