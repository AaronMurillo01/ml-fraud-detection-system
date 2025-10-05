/**
 * Service Worker for FraudGuard AI
 * Provides offline support, caching, and background sync
 */

const CACHE_VERSION = 'v1.0.0';
const CACHE_NAME = `fraudguard-${CACHE_VERSION}`;

// Assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/dashboard',
  '/static/dashboard.html',
  '/static/js/dashboard.js',
  '/static/manifest.json',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js',
];

// API endpoints to cache (with network-first strategy)
const API_CACHE_PATTERNS = [
  '/api/v1/history/stats/summary',
  '/api/v1/history/',
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[Service Worker] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('[Service Worker] Installed successfully');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('[Service Worker] Installation failed:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => name.startsWith('fraudguard-') && name !== CACHE_NAME)
            .map((name) => {
              console.log('[Service Worker] Deleting old cache:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => {
        console.log('[Service Worker] Activated successfully');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip WebSocket requests
  if (url.protocol === 'ws:' || url.protocol === 'wss:') {
    return;
  }
  
  // Skip chrome-extension requests
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  // Handle API requests with network-first strategy
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirstStrategy(request));
    return;
  }
  
  // Handle static assets with cache-first strategy
  event.respondWith(cacheFirstStrategy(request));
});

/**
 * Cache-first strategy
 * Try cache first, fall back to network
 */
async function cacheFirstStrategy(request) {
  try {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      console.log('[Service Worker] Serving from cache:', request.url);
      return cachedResponse;
    }
    
    console.log('[Service Worker] Fetching from network:', request.url);
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('[Service Worker] Fetch failed:', error);
    
    // Return offline page if available
    const cache = await caches.open(CACHE_NAME);
    const offlinePage = await cache.match('/offline.html');
    
    if (offlinePage) {
      return offlinePage;
    }
    
    // Return a basic offline response
    return new Response('Offline - Please check your internet connection', {
      status: 503,
      statusText: 'Service Unavailable',
      headers: new Headers({
        'Content-Type': 'text/plain',
      }),
    });
  }
}

/**
 * Network-first strategy
 * Try network first, fall back to cache
 */
async function networkFirstStrategy(request) {
  try {
    console.log('[Service Worker] Fetching from network (API):', request.url);
    const networkResponse = await fetch(request);
    
    // Cache successful GET responses
    if (networkResponse.ok && request.method === 'GET') {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('[Service Worker] Network fetch failed, trying cache:', error);
    
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      console.log('[Service Worker] Serving API from cache:', request.url);
      return cachedResponse;
    }
    
    // Return error response
    return new Response(
      JSON.stringify({
        error: 'Offline',
        message: 'Unable to fetch data. Please check your internet connection.',
      }),
      {
        status: 503,
        statusText: 'Service Unavailable',
        headers: new Headers({
          'Content-Type': 'application/json',
        }),
      }
    );
  }
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  console.log('[Service Worker] Background sync:', event.tag);
  
  if (event.tag === 'sync-transactions') {
    event.waitUntil(syncTransactions());
  }
});

/**
 * Sync pending transactions when back online
 */
async function syncTransactions() {
  try {
    // Get pending transactions from IndexedDB or localStorage
    // This is a placeholder - implement actual sync logic
    console.log('[Service Worker] Syncing pending transactions...');
    
    // Example: Send pending transactions to server
    // const pendingTransactions = await getPendingTransactions();
    // for (const transaction of pendingTransactions) {
    //   await fetch('/api/v1/fraud/analyze', {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify(transaction),
    //   });
    // }
    
    console.log('[Service Worker] Sync completed');
  } catch (error) {
    console.error('[Service Worker] Sync failed:', error);
    throw error; // Retry sync
  }
}

// Push notifications
self.addEventListener('push', (event) => {
  console.log('[Service Worker] Push notification received');
  
  const options = {
    body: event.data ? event.data.text() : 'New fraud alert',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    vibrate: [200, 100, 200],
    tag: 'fraud-alert',
    requireInteraction: true,
    actions: [
      {
        action: 'view',
        title: 'View Details',
        icon: '/static/icons/action-view.png',
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: '/static/icons/action-dismiss.png',
      },
    ],
  };
  
  event.waitUntil(
    self.registration.showNotification('FraudGuard Alert', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  console.log('[Service Worker] Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow('/dashboard')
    );
  }
});

// Message handler for communication with clients
self.addEventListener('message', (event) => {
  console.log('[Service Worker] Message received:', event.data);
  
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data.type === 'CACHE_URLS') {
    event.waitUntil(
      caches.open(CACHE_NAME)
        .then((cache) => cache.addAll(event.data.urls))
    );
  }
  
  if (event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.delete(CACHE_NAME)
        .then(() => {
          console.log('[Service Worker] Cache cleared');
        })
    );
  }
});

// Periodic background sync (if supported)
self.addEventListener('periodicsync', (event) => {
  console.log('[Service Worker] Periodic sync:', event.tag);
  
  if (event.tag === 'update-stats') {
    event.waitUntil(updateStats());
  }
});

/**
 * Update statistics in background
 */
async function updateStats() {
  try {
    console.log('[Service Worker] Updating stats...');
    
    const response = await fetch('/api/v1/history/stats/summary?days=30');
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put('/api/v1/history/stats/summary?days=30', response.clone());
      console.log('[Service Worker] Stats updated');
    }
  } catch (error) {
    console.error('[Service Worker] Stats update failed:', error);
  }
}

console.log('[Service Worker] Loaded');

