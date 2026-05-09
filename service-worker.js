// A simple Service Worker to cache the application's core assets
const CACHE_NAME = 'marketpulse-v3';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './data.json',
  './final_app_icon.png',
  './manifest.json'
];

// Installation event: Cache the required assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Opened cache and adding assets');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

// Activation event: Clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Fetch event: Serve assets from the cache if available
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      // If the asset is in the cache, return it
      if (response) {
        return response;
      }
      // Otherwise, fetch it from the network
      return fetch(event.request).then((networkResponse) => {
        // If the fetch fails, or the response isn't good, return the network response
        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
          return networkResponse;
        }
        // If the fetch succeeds, clone the response and save it to the cache
        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });
        return networkResponse;
      });
    })
  );
});
