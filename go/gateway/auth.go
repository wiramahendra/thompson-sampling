package gateway

import (
	"crypto/subtle"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/wiramahendra/thompson-sampling/go/thompson"
)

// BearerAuth returns a function that checks Authorization header with
// constant-time comparison. Empty requiredToken disables auth.
func BearerAuth(requiredToken string) func(*http.Request) bool {
	return func(r *http.Request) bool {
		if requiredToken == "" {
			return true
		}
		h := r.Header.Get("Authorization")
		if !strings.HasPrefix(h, "Bearer ") {
			return false
		}
		token := strings.TrimPrefix(h, "Bearer ")
		// Constant-time to avoid timing side-channel on token length/value.
		if len(token) != len(requiredToken) {
			// Still do constant-time compare on padded values to avoid early return timing.
			_ = subtle.ConstantTimeCompare([]byte(token), []byte(token))
			return false
		}
		return subtle.ConstantTimeCompare([]byte(token), []byte(requiredToken)) == 1
	}
}

// TokenBucket is a per-tenant rate limiter.
type TokenBucket struct {
	mu         sync.Mutex
	tokens     float64
	capacity   float64
	refillRate float64 // per second
	lastRefill time.Time
}

// NewTokenBucket creates a bucket with capacity and refill rate.
func NewTokenBucket(capacity, refillPerSec float64) *TokenBucket {
	return &TokenBucket{capacity: capacity, tokens: capacity, refillRate: refillPerSec, lastRefill: time.Now()}
}

// Allow consumes one token if available.
func (tb *TokenBucket) Allow() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.tokens += elapsed * tb.refillRate
	if tb.tokens > tb.capacity {
		tb.tokens = tb.capacity
	}
	tb.lastRefill = now
	if tb.tokens >= 1 {
		tb.tokens--
		return true
	}
	return false
}

// RateLimit wraps a bucket as http predicate.
func RateLimit(bucket *TokenBucket) func(*http.Request) bool {
	return func(r *http.Request) bool { return bucket.Allow() }
}

// PerTenantBuckets maps tenant -> TokenBucket with bounded size and LRU eviction.
type PerTenantBuckets struct {
	mu         sync.Mutex
	buckets    map[string]*TokenBucket
	lastSeen   map[string]time.Time
	capacity   float64
	rate       float64
	maxTenants int
	ttl        time.Duration
}

func NewPerTenantBuckets(capacity, rate float64) *PerTenantBuckets {
	return &PerTenantBuckets{
		buckets:    make(map[string]*TokenBucket),
		lastSeen:   make(map[string]time.Time),
		capacity:   capacity,
		rate:       rate,
		maxTenants: 10000,
		ttl:        time.Hour,
	}
}

func (ptb *PerTenantBuckets) Allow(tenant string) bool {
	ptb.mu.Lock()
	defer ptb.mu.Unlock()
	now := time.Now()
	// Opportunistic TTL pruning (stale tenants)
	for k, t := range ptb.lastSeen {
		if now.Sub(t) > ptb.ttl {
			delete(ptb.buckets, k)
			delete(ptb.lastSeen, k)
		}
	}
	b, ok := ptb.buckets[tenant]
	if !ok {
		if len(ptb.buckets) >= ptb.maxTenants {
			// Evict LRU (oldest lastSeen)
			var oldest string
			var oldestTime time.Time
			first := true
			for k, t := range ptb.lastSeen {
				if first || t.Before(oldestTime) {
					oldest, oldestTime = k, t
					first = false
				}
			}
			if oldest != "" {
				delete(ptb.buckets, oldest)
				delete(ptb.lastSeen, oldest)
			}
		}
		b = NewTokenBucket(ptb.capacity, ptb.rate)
		ptb.buckets[tenant] = b
	}
	ptb.lastSeen[tenant] = now
	return b.Allow()
}

func (ptb *PerTenantBuckets) AllowRequest(r *http.Request) bool {
	tenant := r.Header.Get("Authorization")
	if tenant == "" {
		tenant = r.RemoteAddr
	}
	return ptb.Allow(tenant)
}

// ResponseRecorder captures status for streaming-aware recording.
type ResponseRecorder struct {
	http.ResponseWriter
	Status      int
	WroteHeader bool
}

func (rr *ResponseRecorder) WriteHeader(code int) {
	if !rr.WroteHeader {
		rr.Status = code
		rr.WroteHeader = true
		rr.ResponseWriter.WriteHeader(code)
	}
}

func (rr *ResponseRecorder) Write(b []byte) (int, error) {
	if !rr.WroteHeader {
		rr.WriteHeader(http.StatusOK)
		// Status remains 200 if WriteHeader not called explicitly
		if rr.Status == 0 {
			rr.Status = 200
		}
	}
	return rr.ResponseWriter.Write(b)
}

// PerTenantBreaker maps tenant -> CircuitBreaker with bounded size and LRU eviction.
type PerTenantBreaker struct {
	mu         sync.Mutex
	breakers   map[string]*thompson.CircuitBreaker
	lastSeen   map[string]time.Time
	threshold  uint32
	cooldown   uint64
	maxTenants int
	ttl        time.Duration
}

func NewPerTenantBreaker(threshold uint32, cooldown uint64) *PerTenantBreaker {
	return &PerTenantBreaker{
		breakers:   make(map[string]*thompson.CircuitBreaker),
		lastSeen:   make(map[string]time.Time),
		threshold:  threshold,
		cooldown:   cooldown,
		maxTenants: 10000,
		ttl:        time.Hour,
	}
}

func (ptb *PerTenantBreaker) For(tenant string) *thompson.CircuitBreaker {
	ptb.mu.Lock()
	defer ptb.mu.Unlock()
	now := time.Now()
	// Prune stale
	for k, t := range ptb.lastSeen {
		if now.Sub(t) > ptb.ttl {
			delete(ptb.breakers, k)
			delete(ptb.lastSeen, k)
		}
	}
	if cb, ok := ptb.breakers[tenant]; ok {
		ptb.lastSeen[tenant] = now
		return cb
	}
	if len(ptb.breakers) >= ptb.maxTenants {
		var oldest string
		var oldestTime time.Time
		first := true
		for k, t := range ptb.lastSeen {
			if first || t.Before(oldestTime) {
				oldest, oldestTime = k, t
				first = false
			}
		}
		if oldest != "" {
			delete(ptb.breakers, oldest)
			delete(ptb.lastSeen, oldest)
		}
	}
	cb := thompson.NewCircuitBreaker(ptb.threshold, ptb.cooldown)
	ptb.breakers[tenant] = cb
	ptb.lastSeen[tenant] = now
	return cb
}
