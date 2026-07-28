import 'package:http/http.dart' as http;
import 'package:xml/xml.dart';

import '../core/skill.dart';

/// Top headlines from RSS feeds (BBC World + Hacker News), parsed on-device.
class NewsSkill extends Skill {
  static const _feeds = [
    'https://feeds.bbci.co.uk/news/world/rss.xml',
    'https://hnrss.org/frontpage',
  ];

  @override
  String get name => 'news';

  @override
  List<RegExp> get patterns => [
        RegExp(r'\b(?:the\s+)?news\b'),
        RegExp(r'\bheadlines?\b'),
        RegExp(r"\bwhat(?:'?s| is)\s+happening\b"),
        RegExp(r'\b(?:world|global|international)\s+news\b'),
        RegExp(r'\bwhat(?:'
            r"'?s| is)\s+going\s+on\s+in\s+the\s+world\b"),
        RegExp(r'\bcurrent\s+events\b'),
      ];

  @override
  Future<SkillResult> run(SkillRequest request) async {
    try {
      final r = await http
          .get(Uri.parse(_feeds.first))
          .timeout(const Duration(seconds: 8));
      if (r.statusCode >= 400) throw Exception('bad status');
      final doc = XmlDocument.parse(r.body);
      final titles = doc
          .findAllElements('item')
          .take(4)
          .map((item) => item.getElement('title')?.innerText.trim() ?? '')
          .where((t) => t.isNotEmpty)
          .toList();
      if (titles.isEmpty) return const SkillResult('No headlines right now.');
      final numbered =
          titles.asMap().entries.map((e) => '${e.key + 1}. ${e.value}').join('. ');
      return SkillResult('Top headlines. $numbered.');
    } catch (_) {
      return const SkillResult(
          "I couldn't fetch the news — are you online?", success: false);
    }
  }
}
