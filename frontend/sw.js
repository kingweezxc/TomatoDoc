// TomatoDoc service worker: keeps the app opening offline.
// Change the version number below whenever you update index.html so phones fetch the new copy.
const CACHE = 'tomatodoc-v2';
const SHELL = ['/', '/manifest.webmanifest', '/icons/pwa-192x192.png', '/icons/pwa-512x512.png']

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)))
  self.skipWaiting()
})

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key)))
    )
  )
  self.clients.claim()
})

self.addEventListener('fetch', event => {
  const req = event.request
  const url = new URL(req.url)
  // Only handle same-site GET requests. Predictions and health checks always go to the server.
  if (req.method !== 'GET' || url.origin !== self.location.origin || url.pathname.startsWith('/api/')) return

  event.respondWith(
    fetch(req)
      .then(res => {
        if (res.ok) {
          const copy = res.clone()
          caches.open(CACHE).then(cache => cache.put(req, copy))
        }
        return res
      })
      .catch(() =>
        caches.match(req).then(hit => hit || (req.mode === 'navigate' ? caches.match('/') : Response.error()))
      )
  )
})
