// Skill routing/behavior tests — pure logic, no widgets or platform channels.
import 'package:flutter_test/flutter_test.dart';
import 'package:medo_phone/core/skill.dart';
import 'package:medo_phone/skills/calc_skill.dart';
import 'package:medo_phone/skills/datetime_skill.dart';
import 'package:medo_phone/skills/weather_skill.dart';
import 'package:medo_phone/skills/websearch_skill.dart';

void main() {
  test('datetime matches the usual phrasings', () {
    final s = DateTimeSkill();
    for (final q in ['what time is it', "what's the time", 'what day is it']) {
      expect(s.matchOf(q), isNotNull, reason: q);
    }
  });

  test('calc evaluates arithmetic with precedence', () async {
    final s = CalcSkill();
    final m = s.matchOf('what is 2 + 3 x 4');
    expect(m, isNotNull);
    final r = await s.run(SkillRequest('what is 2 + 3 x 4', m!));
    expect(r.speech, contains('14')); // 2 + (3*4)
  });

  test('calc converts units offline', () async {
    final s = CalcSkill();
    final m = s.matchOf('convert 10 km to miles');
    expect(m, isNotNull);
    final r = await s.run(SkillRequest('convert 10 km to miles', m!));
    expect(r.speech.toLowerCase(), contains('miles'));
  });

  test('weather captures a named city', () {
    final s = WeatherSkill();
    final m = s.matchOf('weather in london');
    expect(m?.namedGroup('city')?.trim(), 'london');
  });

  test('web search is the broad fallback, not stealing the clock', () {
    expect(DateTimeSkill().matchOf('what is the time'), isNotNull);
    expect(WebSearchSkill().matchOf('what is a black hole'), isNotNull);
  });
}
