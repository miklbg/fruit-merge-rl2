// Service Worker for Fruit Merge Game
// Update this version number whenever you make changes to force cache refresh
const VERSION = '1.0.0';
const CACHE_NAME = `fruit-merge-v${VERSION}`;
const urlsToCache = [
  './index.html',
  './css/tailwind.min.css',
  './matter.min.js',
  './audio/web-audio-bgm.js',
  './assets/icon.png',
  './assets/world.png',
  './assets/1-blueberry.png',
  './assets/2-strawberry.png',
  './assets/3-lemon.png',
  './assets/4-apple.png',
  './assets/5-orange.png',
  './assets/6-grapes.png',
  './assets/7-cantaloupe.png',
  './assets/8-coconut.png',
  './assets/9-pineapple.png',
  './assets/10-watermelon.png',
  './assets/children-music-loop-creative-fun-262427.mp3',
  './manifest.json'
];

/**
 * Checks if a response is valid for caching
 * @param {Response} response - The fetch response to validate
 * @returns {boolean} True if response should be cached (status 200-299 and type 'basic' or 'cors')
 */
function isValidResponse(response) {
  return response && 
         response.status >= 200 && 
         response.status < 300 && 
         (response.type === 'basic' || response.type === 'cors');
}

// Install event - cache resources
self.addEventListener('install', event => {
  // Force the waiting service worker to become the active service worker
  self.skipWaiting();
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache:', CACHE_NAME);
        return cache.addAll(urlsToCache);
      })
      .catch(err => {
        console.error('Cache installation failed:', err);
      })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  
  // For HTML files, use network-first strategy to ensure updates
  if (event.request.destination === 'document' || url.pathname.endsWith('.html')) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // Clone and cache the fresh response
          if (isValidResponse(response)) {
            const responseToCache = response.clone();
            caches.open(CACHE_NAME).then(cache => {
              cache.put(event.request, responseToCache);
            });
          }
          return response;
        })
        .catch(() => {
          // If network fails, fallback to cache
          return caches.match(event.request);
        })
    );
    return;
  }
  
  // For other resources, use cache-first strategy
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }
        
        // Clone the request
        const fetchRequest = event.request.clone();
        
        return fetch(fetchRequest).then(response => {
          // Check if valid response for caching
          if (!isValidResponse(response)) {
            return response;
          }
          
          // Clone the response
          const responseToCache = response.clone();
          
          // Cache the response asynchronously
          caches.open(CACHE_NAME)
            .then(cache => {
              return cache.put(event.request, responseToCache);
            })
            .catch(err => {
              console.error('Failed to cache response:', err);
            });
          
          return response;
        }).catch(err => {
          console.error('Fetch failed:', err);
          // Network request failed - the resource is not cached
          throw err;
        });
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  // Take control of all clients immediately
  event.waitUntil(
    clients.claim().then(() => {
      const cacheWhitelist = [CACHE_NAME];
      return caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (!cacheWhitelist.includes(cacheName)) {
              console.log('Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      });
    }).then(() => {
      // Notify all clients that a new version is active
      return self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'SW_UPDATED',
            version: VERSION
          });
        });
      });
    })
  );
});

// Message handler for manual updates
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
