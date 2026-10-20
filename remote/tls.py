"""HTTPS for the companion API: a self-signed certificate, made on first run.

The API is LAN-only and bearer-token authenticated, but over plain HTTP the
token crosses the network in the clear — anyone on the same Wi-Fi can read it
off a watch's request and then drive MEDO. TLS closes that, and the only
certificate that makes sense for a machine with no domain name is one MEDO
mints for itself.

What that does and does not buy you, stated plainly because self-signed
certificates are routinely oversold:

* It encrypts the channel, so the token is no longer readable on the wire.
  That is the actual threat here and it is fully solved.
* It does NOT prove identity to a client that has never seen this machine
  before — no CA vouches for it. A LAN attacker who can already redirect your
  traffic could present their own self-signed certificate instead. Clients that
  pin the fingerprint (printed on first run, and served at ``/tls``) get
  identity too; clients that blindly trust any certificate get only privacy.

The certificate covers ``localhost``, every local IPv4 address found on the
machine, and the hostname, so the HUD on 127.0.0.1 and the watch on
192.168.x.y both match the same file. It is regenerated when it expires or when
the machine's addresses change enough that the old one no longer covers them.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import ipaddress
import logging
import os
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

#: How long a freshly minted certificate is good for. 825 days is the CA/Browser
#: Forum's cap for publicly-trusted leaf certificates; matching it keeps the file
#: acceptable to clients that enforce that limit on private certificates too.
VALIDITY_DAYS = 825

#: Regenerate this long before expiry, so a long-running install rolls over on a
#: restart rather than dying mid-week with an expired certificate.
RENEW_BEFORE_DAYS = 30

CERT_NAME = "medo-cert.pem"
KEY_NAME = "medo-key.pem"


def local_hostnames() -> list[str]:
    """DNS names this machine answers to (``localhost`` first, always)."""
    names = ["localhost"]
    with contextlib.suppress(Exception):
        host = socket.gethostname()
        if host and host not in names:
            names.append(host)
        # The mDNS name a phone on the LAN is most likely to use.
        if host and f"{host}.local" not in names:
            names.append(f"{host}.local")
    return names


def local_addresses() -> list[str]:
    """Every IPv4/IPv6 literal this machine can be reached on, loopback first.

    Best-effort by design: an interface enumeration that fails must not stop
    MEDO from serving, it just yields a certificate covering fewer addresses.
    """
    found = ["127.0.0.1", "::1"]
    with contextlib.suppress(Exception):
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            # Strip a link-local zone id ("fe80::1%eth0") — not a valid literal.
            addr = addr.split("%", 1)[0]
            if addr not in found:
                found.append(addr)
    return found


def _san_entries(hostnames: list[str], addresses: list[str]):
    from cryptography import x509

    entries: list = [x509.DNSName(h) for h in hostnames]
    for addr in addresses:
        with contextlib.suppress(ValueError):
            entries.append(x509.IPAddress(ipaddress.ip_address(addr)))
    return entries


def fingerprint(cert_path: str | Path) -> str:
    """The certificate's SHA-256 fingerprint, colon-separated and uppercase.

    This is what a client pins to get identity as well as privacy, so it is
    printed on first run and served (to authenticated callers) at ``/tls``.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes

    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    digest = cert.fingerprint(hashes.SHA256())
    return ":".join(f"{b:02X}" for b in digest)


def _covers(cert_path: Path, hostnames: list[str], addresses: list[str]) -> bool:
    """True when an existing certificate is still usable as-is.

    Usable means: parses, is not within :data:`RENEW_BEFORE_DAYS` of expiry, and
    its SANs cover everything we would put in a fresh one. The last check is
    what makes a laptop that moved to a new network re-mint automatically
    instead of serving a certificate that matches none of its addresses.
    """
    from cryptography import x509

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    except Exception:
        return False
    now = _dt.datetime.now(_dt.timezone.utc)
    not_after = cert.not_valid_after_utc
    if not_after - now < _dt.timedelta(days=RENEW_BEFORE_DAYS):
        return False
    try:
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return False
    have_names = set(san.get_values_for_type(x509.DNSName))
    have_ips = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
    if not set(hostnames) <= have_names:
        return False
    wanted_ips = set()
    for addr in addresses:
        with contextlib.suppress(ValueError):
            wanted_ips.add(str(ipaddress.ip_address(addr)))
    return wanted_ips <= have_ips


def generate_certificate(cert_path: Path, key_path: Path, *,
                         hostnames: list[str] | None = None,
                         addresses: list[str] | None = None,
                         common_name: str = "MEDO") -> tuple[Path, Path]:
    """Write a fresh self-signed certificate + private key. Returns both paths."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    hostnames = hostnames or local_hostnames()
    addresses = addresses or local_addresses()
    # P-256 rather than RSA: a fraction of the generation time (this runs on the
    # first launch, in front of a waiting user) and universally supported by
    # every TLS client that matters here.
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MEDO (self-signed)"),
    ])
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # Backdated an hour so a client whose clock runs slightly behind this
        # machine's doesn't reject a certificate minted seconds ago.
        .not_valid_before(now - _dt.timedelta(hours=1))
        .not_valid_after(now + _dt.timedelta(days=VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(
            _san_entries(hostnames, addresses)), critical=False)
        # CA:TRUE on a leaf looks wrong until you remember what this file is
        # for: clients trust it DIRECTLY (pinned as a CA file, or installed in
        # the OS store on a phone), and OpenSSL will not accept a non-CA
        # certificate as the anchor of its own chain — it reports "unable to get
        # local issuer certificate" instead. This is the shape `openssl req
        # -x509` produces for exactly this reason. The key never leaves this
        # machine, so the signing power it nominally confers is MEDO's alone.
        .add_extension(x509.BasicConstraints(ca=True, path_length=0),
                       critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=True, key_encipherment=False,
            key_cert_sign=True, key_agreement=True, content_commitment=False,
            data_encipherment=False, crl_sign=False,
            encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage(
            [x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
                       critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    # Key first, and created with 0600 already applied — writing it world-readable
    # and chmod-ing after leaves a window where any local user can copy it.
    key_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with contextlib.suppress(OSError):
        key_path.unlink()
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key_bytes)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    logger.info("minted a self-signed TLS certificate at %s (valid %d days)",
                cert_path, VALIDITY_DAYS)
    return cert_path, key_path


def ensure_certificate(cert_dir: str | Path, *,
                       certfile: str | Path | None = None,
                       keyfile: str | Path | None = None,
                       hostnames: list[str] | None = None,
                       addresses: list[str] | None = None) -> tuple[Path, Path]:
    """The certificate + key to serve with, minting them if needed.

    An operator-supplied ``certfile``/``keyfile`` pair (a real certificate from
    their own CA) is used verbatim and never regenerated — MEDO only manages the
    self-signed one it made itself.
    """
    if certfile and keyfile:
        cert_path, key_path = Path(certfile), Path(keyfile)
        if not cert_path.exists() or not key_path.exists():
            raise FileNotFoundError(
                f"remote.tls_certfile/tls_keyfile point at files that don't "
                f"exist: {cert_path}, {key_path}")
        return cert_path, key_path

    directory = Path(cert_dir)
    cert_path, key_path = directory / CERT_NAME, directory / KEY_NAME
    hostnames = hostnames or local_hostnames()
    addresses = addresses or local_addresses()
    if cert_path.exists() and key_path.exists() and _covers(
            cert_path, hostnames, addresses):
        return cert_path, key_path
    return generate_certificate(cert_path, key_path,
                                hostnames=hostnames, addresses=addresses)


def ssl_context(cert_path: str | Path, key_path: str | Path):
    """A server-side :mod:`ssl` context for ``aiohttp``'s ``TCPSite``."""
    import ssl

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # TLS 1.2 floor: everything below it is broken, and every client MEDO talks
    # to (browsers, watchOS/Android HTTP stacks, httpx) speaks at least 1.2.
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx


def _amain() -> None:  # pragma: no cover - operator utility
    """``python -m remote.tls`` — mint/inspect the certificate and print its
    fingerprint, so a client can be pinned without starting the server."""
    from core.config import PROJECT_ROOT

    cert, key = ensure_certificate(PROJECT_ROOT / "data" / "tls")
    print(f"certificate : {cert}")
    print(f"private key : {key}")
    print(f"SHA-256     : {fingerprint(cert)}")
    print(f"hostnames   : {', '.join(local_hostnames())}")
    print(f"addresses   : {', '.join(local_addresses())}")


if __name__ == "__main__":  # pragma: no cover
    _amain()


def _digest_of(path: str | Path) -> str:
    """SHA-256 of a file's bytes — used by tests to prove a cert was reused."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
