import 'package:intl/intl.dart';

import '../core/skill.dart';

/// Time and date, straight from the phone clock.
class DateTimeSkill extends Skill {
  @override
  String get name => 'time';

  @override
  List<RegExp> get patterns => [
        RegExp(r'\bwhat(?:'
            r"'?s| is)?\s+the\s+time\b"),
        RegExp(r'\bwhat\s+time\s+is\s+it\b'),
        RegExp(r'\b(?:current\s+)?time\b'),
        RegExp(r"\bwhat(?:'?s| is)?\s+(?:the\s+)?date\b"),
        RegExp(r'\bwhat\s+day\s+is\s+it\b'),
        RegExp(r"\bwhat(?:'?s| is)?\s+today\b"),
      ];

  @override
  Future<SkillResult> run(SkillRequest request) async {
    final now = DateTime.now();
    final t = request.text;
    if (t.contains('date') || t.contains('day') || t.contains('today')) {
      return SkillResult("It's ${DateFormat('EEEE, MMMM d').format(now)}.");
    }
    return SkillResult("It's ${DateFormat('h:mm a').format(now)}.");
  }
}
