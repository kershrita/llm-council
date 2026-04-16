const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');

const PORT = Number(process.env.PORT || '80');
const API_TARGET_HOST = process.env.API_TARGET_HOST || 'backend';
const API_TARGET_PORT = Number(process.env.API_TARGET_PORT || '8001');
const DIST_DIR = path.join(__dirname, 'dist');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
};

function sendFile(res, filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';
  const stream = fs.createReadStream(filePath);

  stream.on('open', () => {
    res.writeHead(200, {
      'Content-Type': contentType,
      'Cache-Control': 'public, max-age=300',
    });
    stream.pipe(res);
  });

  stream.on('error', () => {
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
    }
    res.end('Failed to read file');
  });
}

function resolveStaticPath(requestUrl) {
  const pathname = decodeURIComponent(requestUrl.split('?')[0]);
  const relativePath = pathname === '/' ? '/index.html' : pathname;
  const absolutePath = path.join(DIST_DIR, relativePath);

  if (!absolutePath.startsWith(DIST_DIR)) {
    return null;
  }

  return absolutePath;
}

function proxyApiRequest(req, res) {
  const proxyReq = http.request(
    {
      hostname: API_TARGET_HOST,
      port: API_TARGET_PORT,
      method: req.method,
      path: req.url,
      headers: {
        ...req.headers,
        host: `${API_TARGET_HOST}:${API_TARGET_PORT}`,
      },
    },
    (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 502, proxyRes.headers);
      proxyRes.pipe(res, { end: true });
    }
  );

  proxyReq.on('error', (error) => {
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'application/json; charset=utf-8' });
    }
    res.end(
      JSON.stringify({
        detail: `API proxy request failed: ${error.message}`,
      })
    );
  });

  req.pipe(proxyReq, { end: true });
}

const server = http.createServer((req, res) => {
  if (!req.url) {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Bad Request');
    return;
  }

  if (req.url.startsWith('/api/')) {
    proxyApiRequest(req, res);
    return;
  }

  const requestedFile = resolveStaticPath(req.url);
  if (!requestedFile) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Forbidden');
    return;
  }

  fs.stat(requestedFile, (err, stat) => {
    if (!err && stat.isFile()) {
      sendFile(res, requestedFile);
      return;
    }

    sendFile(res, path.join(DIST_DIR, 'index.html'));
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(
    `Frontend server running on 0.0.0.0:${PORT} (proxying /api to ${API_TARGET_HOST}:${API_TARGET_PORT})`
  );
});
