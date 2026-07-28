"""The vision sidecar's request guards (webcam + pointer are unauthenticated, so
the Host/cross-site checks are the only thing between a drive-by page and the
camera). Pure unit test of the loopback-Host rule — no socket, no camera."""

from __future__ import annotations

from vision.run import _host_is_loopback


def test_host_is_loopback_accepts_localhost_refuses_domains_and_lan_ips():
    for ok in ("127.0.0.1:8731", "localhost:8731", "127.0.0.1", "[::1]:8731", ""):
        assert _host_is_loopback(ok) is True, ok
    # A domain Host is a DNS-rebinding attempt; a LAN IP can't reach the loopback
    # bind legitimately — both refused.
    for bad in ("evil.example:8731", "192.168.1.5:8731", "medo.local"):
        assert _host_is_loopback(bad) is False, bad
