import 'dart:convert';

import 'package:http/http.dart' as http;

import '../core/skill.dart';

/// Keyless web answers via DuckDuckGo's Instant Answer API. Great for facts
/// with a definition/abstract; falls back to opening the query in the browser
/// (handled by the caller) when there's no instant answer.
class WebSearchSkill extends Skill {
  @override
  String get name => 'web_search';

  @override
  List<RegExp> get patterns => [
        RegExp(r'\b(?:search|google|look\s+up)\s+(?:the\s+web\s+for\s+|for\s+)?(?<q>.+)'),
        RegExp(r'\bwhat\s+is\s+the\s+latest\s+(?:on|about)\s+(?<q2>.+)'),
        RegExp(r'\bwho\s+(?:is|was)\s+(?<q3>.+)'),
        RegExp(r'\bwhat\s+is\s+(?:a\s+)?(?<q4>.+)'),
      ];

  @override
  Future<SkillResult> run(SkillRequest request) async {
    final m = request.match;
    final q = [
      for (final g in ['q', 'q2', 'q3', 'q4'])
        if (m.groupNames.contains(g)) m.namedGroup(g)
    ].firstWhere((s) => s != null && s.trim().isNotEmpty, orElse: () => null)?.trim();
    if (q == null || q.isEmpty) {
      return const SkillResult('What should I look up?', success: false);
    }
    try {
      final r = await http
          .get(Uri.parse(
              'https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q=${Uri.encodeQueryComponent(q)}'))
          .timeout(const Duration(seconds: 8));
      final body = jsonDecode(r.body) as Map<String, Object?>;
      final abstract = (body['AbstractText'] as String?)?.trim();
      final answer = (body['Answer'] as String?)?.trim();
      final definition = (body['Definition'] as String?)?.trim();
      final best = [answer, abstract, definition]
          .firstWhere((s) => s != null && s.isNotEmpty, orElse: () => null);
      if (best == null) {
        return SkillResult("I didn't find a quick answer for $q.",
            success: false, data: {'openWeb': q});
      }
      // Keep it speakable — first two sentences.
      final sentences = best.split(RegExp(r'(?<=[.!?])\s+')).take(2).join(' ');
      return SkillResult(sentences);
    } catch (_) {
      return const SkillResult("I couldn't reach the web — are you online?",
          success: false);
    }
  }
}
