/**
 * cam-playlist — Supabase Edge Function
 *
 * Lets the dashboard play Surfline cams live. Surfline's CDN only serves the
 * HLS playlist to requests whose Origin is surfline.com, which a browser on
 * our dashboard can't send. The video segments themselves are open (CORS *),
 * so this function fetches just the playlist (~1 KB) and rewrites its segment
 * lines to absolute CDN URLs. The browser then pulls the video straight from
 * Surfline; nothing heavy passes through here.
 *
 * GET /functions/v1/cam-playlist?slug=<hls_slug>[&file=<variant>.m3u8]
 *
 * Deployed with verify_jwt=false: Safari plays HLS natively and can't attach
 * an apikey header. Input is limited to a slug and playlist filename on
 * Surfline's CDN, so this can't be used as an open proxy.
 */

const CDN_BASE = "https://hls.cdn-surfline.com/oregon";

const SURFLINE_HEADERS = {
  "origin": "https://www.surfline.com",
  "referer": "https://www.surfline.com/",
  "user-agent":
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " +
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
};

const SLUG_RE = /^wc-[a-z0-9]+$/;
const FILE_RE = /^[A-Za-z0-9_.-]+\.m3u8$/;

function error(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
  });
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: CORS_HEADERS });
  }

  const url = new URL(req.url);
  const slug = url.searchParams.get("slug") ?? "";
  const file = url.searchParams.get("file") ?? "playlist.m3u8";
  if (!SLUG_RE.test(slug)) return error(400, "valid slug is required");
  if (!FILE_RE.test(file)) return error(400, "invalid file");

  const base = `${CDN_BASE}/${slug}/`;
  let upstream: Response;
  try {
    upstream = await fetch(base + file, { headers: SURFLINE_HEADERS });
  } catch (e) {
    return error(502, String(e).slice(0, 120));
  }
  if (!upstream.ok) {
    await upstream.body?.cancel();
    return error(upstream.status === 404 ? 404 : 502, `upstream HTTP ${upstream.status}`);
  }

  // URI lines: nested playlists route back through this function (they need
  // the Surfline origin too); segments point straight at the CDN.
  // req.url is internal inside the edge runtime, so build the public URL.
  const self = `${Deno.env.get("SUPABASE_URL")}/functions/v1/cam-playlist`;
  const body = (await upstream.text())
    .split("\n")
    .map((line) => {
      const t = line.trim();
      if (!t || t.startsWith("#") || /^https?:\/\//.test(t)) return line;
      if (FILE_RE.test(t)) return `${self}?slug=${slug}&file=${encodeURIComponent(t)}`;
      return base + t;
    })
    .join("\n");

  return new Response(body, {
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/vnd.apple.mpegurl",
      // Live playlists roll every few seconds
      "Cache-Control": "no-store",
    },
  });
});
