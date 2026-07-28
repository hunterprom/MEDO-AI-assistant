import 'package:flutter_test/flutter_test.dart';
import 'package:medo_link/medo_link.dart';

void main() {
  test('MedoReply parses a full server payload', () {
    final reply = MedoReply.fromJson(const {
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

  test('MedoReply tolerates missing optional fields', () {
    final reply = MedoReply.fromJson(const {'speech': 'Hello.', 'path': 'LLM'});
    expect(reply.skill, isNull);
    expect(reply.latencyMs, 0);
  });

  test('MedoException carries a watch-sized message', () {
    const e = MedoException("Can't reach Medo at 1.2.3.4:8710.");
    expect(e.toString(), contains('1.2.3.4'));
  });
}
