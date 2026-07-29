/// HTTP client for MEDO's companion API (`remote/server.py`).
///
/// Tiny surface: `GET /ping` to check the server, `POST /ask {"text": …}` to
/// route one utterance, and `POST /pair/start` + `/pair/confirm` for the
/// presence-code pairing that delivers the auth token. LAN-only, so calls fail
/// fast rather than hang.
library;

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

/// One routed reply from MEDO.
class MedoReply {
  const MedoReply({
    required this.speech,
    required this.path,
    this.skill,
    this.latencyMs = 0,
  });

  /// The text MEDO wants spoken.
  final String speech;

  /// Which brain answered: `FAST` (rule-based skill) or `LLM`/`CHAT`.
  final String path;

  /// Skill name when a fast-path skill handled it.
  final String? skill;

  /// Server-side routing latency (ms).
  final double latencyMs;

  factory MedoReply.fromJson(Map<String, dynamic> json) => MedoReply(
        speech: json['speech'] as String? ?? '',
        path: json['path'] as String? ?? '?',
        skill: json['skill'] as String?,
        latencyMs: (json['latency_ms'] as num? ?? 0).toDouble(),
      );
}

/// Raised for anything that stops us reaching MEDO, with a message short
/// enough to show on a small screen.
class MedoException implements Exception {
  const MedoException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// One poll of an approve-on-PC pairing request.
class PairResult {
  const PairResult(this.status, this.token);

  /// `pending` | `approved` | `denied` | `expired`.
  final String status;

  /// The bearer token — set only when [status] is `approved`.
  final String token;
}

class MedoClient {
  MedoClient(this.address, [this.token = '']);

  /// `host:port` of the machine running `python main.py --serve`.
  final String address;

  /// Bearer token for the companion API (`remote.token` in the server's
  /// secrets.local.yaml). Empty = send nothing; the server only demands it
  /// from LAN clients when auth is enabled.
  final String token;

  static const _pingTimeout = Duration(seconds: 3);
  static const _askTimeout = Duration(seconds: 90); // LLM replies can be slow

  Uri _uri(String path) => Uri.parse('http://$address$path');

  Map<String, String> _headers([Map<String, String> extra = const {}]) => {
        if (token.isNotEmpty) 'Authorization': 'Bearer $token',
        ...extra,
      };

  /// Returns the assistant's name if the server is reachable.
  Future<String> ping() async {
    final body = await _request(
      () => http.get(_uri('/ping'), headers: _headers()).timeout(_pingTimeout),
    );
    return body['name'] as String? ?? 'MEDO';
  }

  /// Sends one utterance and returns MEDO's reply.
  Future<MedoReply> ask(String text) async {
    final body = await _request(
      () => http
          .post(
            _uri('/ask'),
            headers: _headers({'Content-Type': 'application/json'}),
            body: jsonEncode({'text': text}),
          )
          .timeout(_askTimeout),
    );
    return MedoReply.fromJson(body);
  }

  /// Ask the server to show a pairing code on its own screen.
  Future<void> pairStart() async {
    await _request(() => http.post(_uri('/pair/start')).timeout(_pingTimeout));
  }

  /// Exchange the code the user read off the PC for the API token.
  Future<String> pairConfirm(String code) async {
    final body = await _request(
      () => http
          .post(
            _uri('/pair/confirm'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'code': code}),
          )
          .timeout(_pingTimeout),
    );
    return body['token'] as String? ?? '';
  }

  // --- approve-on-PC pairing (no code to type) -------------------------------

  /// Enqueue an approve-on-PC request; returns the `request_id` to poll.
  ///
  /// The user then taps *Approve* in the desktop HUD's Device-fleet panel —
  /// no 6-digit code. [kind] is `phone` or `watch` (shown next to the request).
  Future<String> pairRequest({required String name, String kind = ''}) async {
    final body = await _request(
      () => http
          .post(
            _uri('/pair/request'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'name': name, if (kind.isNotEmpty) 'kind': kind}),
          )
          .timeout(_pingTimeout),
    );
    return body['request_id'] as String? ?? '';
  }

  /// Poll a pairing request once. Token is present only when approved.
  Future<PairResult> pairPoll(String requestId) async {
    final body = await _request(
      () => http
          .get(_uri('/pair/poll?request_id='
              '${Uri.encodeQueryComponent(requestId)}'))
          .timeout(_pingTimeout),
    );
    return PairResult(
      body['status'] as String? ?? 'expired',
      body['token'] as String? ?? '',
    );
  }

  /// The whole approve-on-PC flow: request, then poll until the user approves
  /// on the PC. Calls [onWaiting] once the request is enqueued. Returns the
  /// token, or throws [MedoException] on denial, expiry, or timeout.
  Future<String> connectViaApproval({
    required String name,
    String kind = '',
    Duration timeout = const Duration(minutes: 2),
    Duration pollEvery = const Duration(seconds: 2),
    void Function()? onWaiting,
  }) async {
    final id = await pairRequest(name: name, kind: kind);
    if (id.isEmpty) throw const MedoException('MEDO refused the request.');
    onWaiting?.call();
    final deadline = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(deadline)) {
      await Future<void>.delayed(pollEvery);
      final PairResult r;
      try {
        r = await pairPoll(id);
      } on MedoException {
        // A single slow/dropped poll (server busy, Wi-Fi blip) must not kill the
        // whole 2-minute wait — the user may be about to tap Approve. Keep
        // polling until the deadline; real denial/expiry arrive as a status.
        continue;
      }
      if (r.status == 'approved') {
        if (r.token.isEmpty) {
          throw const MedoException('Approved, but no token came back.');
        }
        return r.token;
      }
      if (r.status == 'denied') {
        throw const MedoException('Connection denied on the PC.');
      }
      if (r.status == 'expired') {
        throw const MedoException('The request expired — try again.');
      }
    }
    throw const MedoException('Timed out waiting for approval on the PC.');
  }

  Future<Map<String, dynamic>> _request(
    Future<http.Response> Function() send,
  ) async {
    late final http.Response response;
    try {
      response = await send();
    } on TimeoutException {
      throw const MedoException('MEDO took too long to answer.');
    } catch (_) {
      throw MedoException("Can't reach MEDO at $address.");
    }
    final Map<String, dynamic> body;
    try {
      body = jsonDecode(response.body) as Map<String, dynamic>;
    } catch (_) {
      throw MedoException('Unexpected reply from $address.');
    }
    if (response.statusCode != 200) {
      throw MedoException(
        body['error'] as String? ?? 'Server error ${response.statusCode}.',
      );
    }
    return body;
  }
}
