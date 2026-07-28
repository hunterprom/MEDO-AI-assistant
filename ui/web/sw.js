/* MEDO HUD service worker — makes the LAN HUD installable + offline-resilient.
 *
 * Deliberately conservative: it only caches the HUD's OWN same-origin shell
 * (the page, manifest, icons). The companion API and the vision MJPEG live on
 * DIFFERENT ports (so a different origin) and the SSE feed is same-origin at
 * /events — all of those are left strictly on the network so live data is
 * never served stale. Navigation is network-first with a cached fallback, so a
 * running server always wins and the app still opens (last shell) when offline.
 *
 * Remove this file + /manifest.json + the <head> tags to fully revert.
 */
const CACHE = 'medo-shell-v1';
const SHELL = [
  '/',
  '/manifest.json',
  '/icons/medo-192.png',
  '/icons/medo-512.png',
  '/icons/medo-maskable-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
      .catch(() => self.skipWaiting())   // a missing asset must not block install
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);
  // Only ever touch same-origin GETs. Cross-origin (the :8710 API, :8731 MJPEG)
  // and non-GET requests fall straight through to the network, untouched.
  if (req.method !== 'GET' || url.origin !== self.location.origin) return;
  // The SSE feed must stream from the live server, never from a cache.
  if (url.pathname === '/events') return;

  // Network-first: fresh when online (the page re-injects its config/token on
  // every load), cached shell as the offline fallback.
  event.respondWith(
    fetch(req)
      .then((resp) => {
        if (resp && resp.ok && resp.type === 'basic') {
          const copy = resp.clone();
          caches.open(CACHE).then((cache) => cache.put(req, copy)).catch(() => {});
        }
        return resp;
      })
      .catch(() => caches.match(req).then((hit) => hit || caches.match('/')))
  );
});
