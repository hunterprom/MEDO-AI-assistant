/// LAN discovery: find the machine running MEDO without typing an IP.
///
/// Broadcasts `MEDO_DISCOVER_V1` over UDP to the companion-API port; the
/// server answers with `{"service": "medo", "name": …, "port": …}` and the
/// datagram's source address tells us where it lives. Pure `dart:io` — no
/// plugins.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// One discovered MEDO server.
class MedoServer {
  const MedoServer({required this.host, required this.port, required this.name});

  final String host;
  final int port;
  final String name;

  String get address => '$host:$port';
}

class MedoDiscovery {
  static const _probe = 'MEDO_DISCOVER_V1';
  // Fixed, well-known discovery port — the server always answers here regardless
  // of its (configurable) API port. The real API port comes from the reply's
  // 'port' field below, so this constant must NOT be confused with the API port.
  static const _port = 8710;

  /// Broadcast and wait for the first MEDO to answer; null when none does.
  ///
  /// Re-sends the probe every second (UDP broadcasts get dropped) until
  /// [timeout] elapses. The reply must parse as JSON with service "medo".
  static Future<MedoServer?> discover({
    Duration timeout = const Duration(seconds: 4),
  }) async {
    final socket = await RawDatagramSocket.bind(InternetAddress.anyIPv4, 0);
    socket.broadcastEnabled = true;
    final completer = Completer<MedoServer?>();

    final sub = socket.listen((event) {
      if (event != RawSocketEvent.read) return;
      final dg = socket.receive();
      if (dg == null || completer.isCompleted) return;
      try {
        final body = jsonDecode(utf8.decode(dg.data)) as Map<String, dynamic>;
        if (body['service'] != 'medo') return;
        completer.complete(MedoServer(
          host: dg.address.address,
          port: (body['port'] as num?)?.toInt() ?? _port,
          name: body['name'] as String? ?? 'MEDO',
        ));
      } catch (_) {
        // Not our reply — keep listening until the timeout.
      }
    });

    final probe = utf8.encode(_probe);
    void send() {
      try {
        socket.send(probe, InternetAddress('255.255.255.255'), _port);
      } catch (_) {
        // Some networks refuse broadcast; the resend timer will retry.
      }
    }

    send();
    final resend = Timer.periodic(const Duration(seconds: 1), (_) => send());
    final result =
        await completer.future.timeout(timeout, onTimeout: () => null);
    resend.cancel();
    await sub.cancel();
    socket.close();
    return result;
  }
}
