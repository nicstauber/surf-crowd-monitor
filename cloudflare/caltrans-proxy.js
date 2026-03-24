/**
 * Caltrans CCTV Proxy — Cloudflare Worker
 *
 * Proxies requests to cwwp2.dot.ca.gov to add CORS headers,
 * enabling browser-side fetch from the CCTV Explorer page.
 *
 * Deploy:
 *   1. wrangler deploy   (or paste into the CF dashboard)
 *
 * Usage from the browser:
 *   fetch(`https://<your-worker>.workers.dev/d07/cctv/cctvStatusD07.json`)
 *
 * The worker strips the leading slash and appends the path to the
 * Caltrans base URL, so any district/path combo works.
 */

const UPSTREAM = 'https://cwwp2.dot.ca.gov/data';

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

export default {
  async fetch(request) {
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (request.method !== 'GET') {
      return new Response('Method not allowed', { status: 405, headers: CORS });
    }

    const url = new URL(request.url);
    // Strip leading slash; forward everything else as-is
    const path = url.pathname.replace(/^\//, '');

    if (!path) {
      return new Response(
        JSON.stringify({ error: 'Provide a path, e.g. /d07/cctv/cctvStatusD07.json' }),
        { status: 400, headers: { 'Content-Type': 'application/json', ...CORS } }
      );
    }

    const upstream = `${UPSTREAM}/${path}`;

    let res;
    try {
      res = await fetch(upstream, {
        headers: { 'User-Agent': 'Mozilla/5.0' },
        cf: { cacheTtl: 30, cacheEverything: true },
      });
    } catch (err) {
      return new Response(
        JSON.stringify({ error: 'Upstream fetch failed', detail: err.message }),
        { status: 502, headers: { 'Content-Type': 'application/json', ...CORS } }
      );
    }

    if (!res.ok) {
      return new Response(
        JSON.stringify({ error: `Upstream returned ${res.status}` }),
        { status: res.status, headers: { 'Content-Type': 'application/json', ...CORS } }
      );
    }

    const body = await res.text();
    return new Response(body, {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=30',
        ...CORS,
      },
    });
  },
};
