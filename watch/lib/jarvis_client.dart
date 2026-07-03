/// HTTP client for the Jarvis v2 companion API (`remote/server.py`).
///
/// The API is tiny: `GET /ping` to check the server is there, `POST /ask`
/// with `{"text": …}` to route one utterance. Both live on the LAN, so
/// every call gets a short timeout — a watch should fail fast, not hang.
library;

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

/// One routed reply from Jarvis.
class JarvisReply {
  const JarvisReply({
    required this.speech,
    required this.path,
    this.skill,
    this.latencyMs = 0,
  });

  /// The text Jarvis wants spoken.
  final String speech;

  /// Which brain answered: `FAST` (rule-based skill) or `LLM` (Ollama).
  final String path;

  /// Skill name when the fast path handled it.
  final String? skill;

  /// Server-side routing latency.
  final double latencyMs;

  factory JarvisReply.fromJson(Map<String, dynamic> json) => JarvisReply(
        speech: json['speech'] as String? ?? '',
        path: json['path'] as String? ?? '?',
        skill: json['skill'] as String?,
        latencyMs: (json['latency_ms'] as num? ?? 0).toDouble(),
      );
}

/// Raised for anything that stops us reaching Jarvis, with a message short
/// enough to show on a watch face.
class JarvisException implements Exception {
  const JarvisException(this.message);
  final String message;

  @override
  String toString() => message;
}

class JarvisClient {
  JarvisClient(this.address);

  /// `host:port` of the machine running `python main.py --serve`.
  final String address;

  static const _pingTimeout = Duration(seconds: 3);
  static const _askTimeout = Duration(seconds: 90); // LLM replies can be slow

  Uri _uri(String path) => Uri.parse('http://$address$path');

  /// Returns the assistant's name if the server is reachable.
  Future<String> ping() async {
    final body = await _request(
      () => http.get(_uri('/ping')).timeout(_pingTimeout),
    );
    return body['name'] as String? ?? 'Jarvis';
  }

  /// Sends one utterance and returns Jarvis's reply.
  Future<JarvisReply> ask(String text) async {
    final body = await _request(
      () => http
          .post(
            _uri('/ask'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'text': text}),
          )
          .timeout(_askTimeout),
    );
    return JarvisReply.fromJson(body);
  }

  Future<Map<String, dynamic>> _request(
    Future<http.Response> Function() send,
  ) async {
    late final http.Response response;
    try {
      response = await send();
    } on TimeoutException {
      throw const JarvisException('Jarvis took too long to answer.');
    } catch (_) {
      throw JarvisException("Can't reach Jarvis at $address.");
    }
    final Map<String, dynamic> body;
    try {
      body = jsonDecode(response.body) as Map<String, dynamic>;
    } catch (_) {
      throw JarvisException('Unexpected reply from $address.');
    }
    if (response.statusCode != 200) {
      throw JarvisException(
        body['error'] as String? ?? 'Server error ${response.statusCode}.',
      );
    }
    return body;
  }
}
