import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

import { getTopNews, LIMITS } from './src/news.js';
import { TOPICS, EDITIONS, DEFAULT_EDITION } from './src/topics.js';

const PORT = Number(process.env.PORT) || 3000;
const PUBLIC_DIR = fileURLToPath(new URL('./public/', import.meta.url));

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.json': 'application/json; charset=utf-8',
};

function sendJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(payload),
    'Cache-Control': 'no-store',
  });
  res.end(payload);
}

async function serveStatic(res, pathname) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Bad request');
    return;
  }

  // normalize() collapses any ../ segments before we join, and the prefix check
  // below is the backstop, so requests cannot escape the public directory.
  const relative = normalize(decoded === '/' ? 'index.html' : decoded).replace(/^(\.\.[/\\])+/, '');
  const filePath = join(PUBLIC_DIR, relative);

  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403).end('Forbidden');
    return;
  }

  try {
    const file = await readFile(filePath);
    res.writeHead(200, {
      'Content-Type': MIME_TYPES[extname(filePath)] ?? 'application/octet-stream',
      'Content-Length': file.length,
    });
    res.end(file);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not found');
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host ?? 'localhost'}`);

  if (req.method !== 'GET') {
    sendJson(res, 405, { error: 'Only GET is supported' });
    return;
  }

  if (url.pathname === '/api/topics') {
    sendJson(res, 200, {
      topics: TOPICS.map(({ id, label, emoji }) => ({ id, label, emoji })),
      editions: Object.entries(EDITIONS).map(([id, { label }]) => ({ id, label })),
      defaults: { n: LIMITS.n.default, days: LIMITS.days.default, edition: DEFAULT_EDITION },
      limits: LIMITS,
    });
    return;
  }

  if (url.pathname === '/api/news') {
    const params = url.searchParams;
    const topics = (params.get('topics') ?? params.get('topic') ?? '')
      .split(',')
      .map((value) => value.trim().toLowerCase())
      .filter(Boolean);
    const query = params.get('q') ?? '';

    if (topics.length === 0 && !query.trim()) {
      sendJson(res, 400, { error: 'Provide ?topics=ai,tech or ?q=your+search' });
      return;
    }

    try {
      const result = await getTopNews({
        topics,
        query,
        n: params.get('n'),
        days: params.get('days'),
        edition: params.get('edition') ?? 'auto',
      });
      sendJson(res, 200, result);
    } catch (error) {
      console.error('[api/news]', error);
      sendJson(res, 502, { error: 'Could not fetch news right now', detail: String(error.message ?? error) });
    }
    return;
  }

  if (url.pathname.startsWith('/api/')) {
    sendJson(res, 404, { error: 'Unknown endpoint' });
    return;
  }

  await serveStatic(res, url.pathname);
});

server.listen(PORT, () => {
  console.log(`topNnews running at http://localhost:${PORT}`);
  console.log(`defaults: n=${LIMITS.n.default}, days=${LIMITS.days.default}`);
});

export { server };
