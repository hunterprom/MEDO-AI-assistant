import '../core/skill.dart';
import '../core/store.dart';

/// Long-term facts: "remember that…", "what do you remember…", "forget…".
class MemorySkill extends Skill {
  @override
  String get name => 'memory';

  @override
  List<RegExp> get patterns => [
        // recall / forget FIRST so "remember" in a question doesn't store it
        RegExp(r'\bwhat\s+do\s+you\s+remember(?:\s+about\s+(?<q>.+))?'),
        RegExp(r'\bforget\s+(?:that\s+|about\s+)?(?<forget>.+)'),
        RegExp(r'^\s*(?:please\s+)?remember\s+(?:that\s+|to\s+)?(?<fact>.+)'),
      ];

  @override
  Future<SkillResult> run(SkillRequest request) async {
    final m = request.match;

    if (m.groupNames.contains('forget') && m.namedGroup('forget') != null) {
      final q = m.namedGroup('forget')!.trim();
      final n = await Store.I.forget(q);
      return SkillResult(n == 0 ? "I don't have anything like that." : 'Forgotten.');
    }
    if (request.text.startsWith('what do you remember')) {
      final q = (m.groupNames.contains('q') ? m.namedGroup('q') : null) ?? '';
      final facts = q.trim().isEmpty ? Store.I.facts() : Store.I.recall(q);
      if (facts.isEmpty) return const SkillResult("I don't remember anything about that yet.");
      return SkillResult(facts.take(4).join('. '));
    }
    // Guard like every other branch: a recall pattern (group 'q', no 'fact')
    // can reach here when the text doesn't start with "what do you remember",
    // and namedGroup('fact') would throw ArgumentError on that match.
    final fact = (m.groupNames.contains('fact') ? m.namedGroup('fact') : null)?.trim() ?? '';
    if (fact.isEmpty) return const SkillResult('Remember what?', success: false);
    final added = await Store.I.addFact(fact);
    return SkillResult(added ? "Got it — I'll remember that." : 'I already knew that.');
  }
}
