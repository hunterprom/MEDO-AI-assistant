"""Cut the network out from under the self-dev gate.

The gate runs ``pytest`` over code a coding agent just wrote, and pytest
executes module-level code at collection time. So agent-authored code runs on
this machine *before* any human has read the diff. The environment is already
scrubbed of credentials and re-homed into a throwaway directory
(``SelfDevEngine._gate_env``); this closes the other half — it must not be able
to reach the network, either to fetch a payload or to post anything out.

This file is never imported by MEDO. It is **copied** into a private directory
that is put first on the gate subprocess's ``PYTHONPATH`` under the name
``sitecustomize``, which CPython imports automatically at interpreter start,
before pytest, before conftest, before any test module. It is kept as a real
file in the repo rather than an exec'd string so it can be linted and tested
like anything else.

**Loopback stays open, deliberately.** MEDO's own suite binds aiohttp test
servers on 127.0.0.1; blocking those would fail every honest proposal and teach
whoever ran it to switch the guard off. What is blocked is everything that
leaves this machine.

**What this is not.** It is an in-process guard, so it binds code that uses the
``socket`` module — which is all of ``httpx``, ``requests``, ``aiohttp``,
``urllib``, and ``asyncio``. It does not bind a subprocess (``curl``), the raw
``_socket`` C type, or a rewritten ``ctypes`` call. Treat it as the layer that
stops accidental and casual egress. For containment against code that is
actively trying to escape, set ``self_dev.sandbox_cmd`` to something with real
OS isolation (``firejail --net=none``, a container) — that is what the setting
is for, and this guard does not replace it.
"""

from __future__ import annotations

import socket as _socket_module

#: Addresses the gate may still reach. Loopback only: the suite's own test
#: servers live here and nothing on this list leaves the machine.
_ALLOWED_HOSTS = frozenset({
    "127.0.0.1", "::1", "localhost", "", "0.0.0.0", "::",
})


class BlockedNetworkAccess(OSError):
    """Raised in place of an outbound connection during a self-dev gate run."""


def _host_of(address) -> str:
    """The host part of a socket address, whatever shape it arrived in."""
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return str(address)


def _is_local(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in _ALLOWED_HOSTS:
        return True
    # 127.0.0.0/8 is all loopback, not just .0.1.
    return host.startswith("127.")


def _refuse(host: str) -> BlockedNetworkAccess:
    return BlockedNetworkAccess(
        f"self-dev gate: outbound network access to {host!r} is blocked. "
        f"Tests run against a proposal may not reach the network.")


def install() -> None:
    """Patch the socket module so only loopback connections succeed."""
    real_connect = _socket_module.socket.connect
    real_connect_ex = _socket_module.socket.connect_ex
    real_create = _socket_module.create_connection
    real_getaddrinfo = _socket_module.getaddrinfo

    def connect(self, address, *args, **kwargs):
        host = _host_of(address)
        if not _is_local(host):
            raise _refuse(host)
        return real_connect(self, address, *args, **kwargs)

    def connect_ex(self, address, *args, **kwargs):
        host = _host_of(address)
        if not _is_local(host):
            raise _refuse(host)
        return real_connect_ex(self, address, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        host = _host_of(address)
        if not _is_local(host):
            raise _refuse(host)
        return real_create(address, *args, **kwargs)

    def getaddrinfo(host, *args, **kwargs):
        # Refusing resolution too, so a name is stopped before a connection is
        # even attempted — and so a library that resolves eagerly fails with
        # this message rather than a confusing DNS error.
        if host is not None and not _is_local(str(host)):
            raise _refuse(str(host))
        return real_getaddrinfo(host, *args, **kwargs)

    _socket_module.socket.connect = connect
    _socket_module.socket.connect_ex = connect_ex
    _socket_module.create_connection = create_connection
    _socket_module.getaddrinfo = getaddrinfo


install()
