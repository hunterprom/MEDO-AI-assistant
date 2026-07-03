import 'package:flutter_test/flutter_test.dart';
import 'package:jarvis_watch/jarvis_client.dart';

void main() {
  test('JarvisReply parses a full server payload', () {
    final reply = JarvisReply.fromJson(const {
      'speech': "It's 10:56 AM.",
      'path': 'FAST',
      'skill': 'datetime',
      'latency_ms': 0.2,
    });
    expect(reply.speech, "It's 10:56 AM.");
    expect(reply.path, 'FAST');
    expect(reply.skill, 'datetime');
    expect(reply.latencyMs, closeTo(0.2, 1e-9));
  });

  test('JarvisReply tolerates missing optional fields', () {
    final reply = JarvisReply.fromJson(const {'speech': 'Hello.', 'path': 'LLM'});
    expect(reply.skill, isNull);
    expect(reply.latencyMs, 0);
  });

  test('JarvisException carries a watch-sized message', () {
    const e = JarvisException("Can't reach Jarvis at 1.2.3.4:8710.");
    expect(e.toString(), contains('1.2.3.4'));
  });
}
