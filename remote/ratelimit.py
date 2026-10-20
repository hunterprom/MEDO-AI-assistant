"""Per-caller request throttling for the companion API.

Bearer-token auth answers *who may call*; it says nothing about *how often*.
Two things need that second answer:

* A stolen or brute-forced token is worth far less if it only buys a few
  requests a second — and an unauthenticated attacker guessing tokens against
  ``/ask`` is reduced from thousands of tries a second to a handful.
* A buggy client (a watch app in a reconnect loop) can otherwise pin a CPU core
  by itself.

A **token bucket** rather than a fixed window: the HUD is bursty by nature —
it fires several polls the moment a tab opens and then goes quiet — and a fixed
window would reject exactly that legitimate burst while happily passing a
steady grind at the limit. Capacity is the burst allowance, the refill rate is
the sustained one.

Everything here is pure and clock-injectable, so the behaviour is unit-tested
without sleeping.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

#: Buckets idle longer than this are forgotten. Without it, an attacker spraying
#: from spoofed source addresses would grow the dict without bound — the limiter
#: itself becomes the memory-exhaustion vector it was added to prevent.
IDLE_EVICT_S = 600.0

#: Hard ceiling on tracked callers. Reached only under an active spray; the
#: oldest-seen buckets are dropped first.
MAX_BUCKETS = 4096


def caller_key(token: str | None, peer: str | None) -> str:
    """Stable identity for throttling: the token if there is one, else the peer.

    The token is **hashed**, never stored: this dict outlives a request, gets
    dumped in tracebacks and metrics, and a secret that never enters it cannot
    leak from it. A truncated digest is plenty — collisions merely make two
    callers share a bucket, they can't grant anyone extra quota.
    """
    if token:
        digest = hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()
        return "tok:" + digest[:16]
    return "ip:" + (peer or "unknown")


@dataclass
class _Bucket:
    tokens: float
    last: float


@dataclass
class RateLimiter:
    """Token-bucket limiter keyed by :func:`caller_key`.

    ``per_minute`` is the sustained rate and ``burst`` the depth. ``burst`` of 0
    means "same as one second of sustained rate", which is almost never what
    anyone wants, so it defaults to a quarter-minute of allowance instead.
    """

    per_minute: float = 240.0
    burst: float = 60.0
    _buckets: dict[str, _Bucket] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.per_minute = max(0.0, float(self.per_minute))
        self.burst = float(self.burst) if self.burst > 0 else self.per_minute / 4

    @property
    def enabled(self) -> bool:
        """A rate of 0 disables throttling entirely (documented escape hatch)."""
        return self.per_minute > 0

    def allow(self, key: str, *, now: float | None = None,
              cost: float = 1.0) -> bool:
        """Spend ``cost`` from ``key``'s bucket. False when it's empty."""
        if not self.enabled:
            return True
        now = time.monotonic() if now is None else now
        bucket = self._buckets.get(key)
        if bucket is None:
            self._evict(now)
            # A first-time caller starts FULL: the very first request from a
            # fresh watch must never be the one that gets thrown away.
            bucket = _Bucket(tokens=self.burst, last=now)
            self._buckets[key] = bucket
        else:
            elapsed = max(0.0, now - bucket.last)
            bucket.tokens = min(self.burst,
                                bucket.tokens + elapsed * self.per_minute / 60.0)
            bucket.last = now
        if bucket.tokens < cost:
            return False
        bucket.tokens -= cost
        return True

    def retry_after(self, key: str, *, now: float | None = None) -> int:
        """Whole seconds until ``key`` can spend one request again (>= 1)."""
        if not self.enabled or self.per_minute <= 0:
            return 1
        bucket = self._buckets.get(key)
        if bucket is None:
            return 1
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - bucket.last)
        available = min(self.burst,
                        bucket.tokens + elapsed * self.per_minute / 60.0)
        missing = max(0.0, 1.0 - available)
        return max(1, int(missing * 60.0 / self.per_minute + 0.999))

    def forget(self, key: str) -> None:
        """Drop a caller's bucket (used when a device is unpaired)."""
        self._buckets.pop(key, None)

    def _evict(self, now: float) -> None:
        stale = [k for k, b in self._buckets.items() if now - b.last > IDLE_EVICT_S]
        for key in stale:
            del self._buckets[key]
        if len(self._buckets) >= MAX_BUCKETS:
            # Oldest-touched first; dict order is insertion order, not recency,
            # so sort rather than trusting it.
            oldest = sorted(self._buckets.items(), key=lambda kv: kv[1].last)
            for key, _ in oldest[: len(self._buckets) - MAX_BUCKETS + 1]:
                del self._buckets[key]
