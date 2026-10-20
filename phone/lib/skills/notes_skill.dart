import '../core/skill.dart';
import '../core/store.dart';

/// Take / read / clear notes, stored locally on the phone.
class NotesSkill extends Skill {
  @override
  String get name => 'notes';

  @override
  List<RegExp> get patterns => [
        RegExp(r'\b(?:take|make|write|jot(?:\s+down)?|add)\s+(?:a\s+)?note\b'
            r'[:\s]*(?<body>.*)'),
        RegExp(r'\bnote\s+that\b\s*(?<body2>.*)'),
        RegExp(r'\b(?:read|list|show|what\s+are)\b.*\bnotes?\b'),
        RegExp(r'\b(?:clear|delete|erase)\s+(?:my\s+)?notes?\b'),
      ];

  @override
  Future<SkillResult> run(SkillRequest request) async {
    final t = request.text;
    final m = request.match;

    if (RegExp(r'\b(?:clear|delete|erase)\b').hasMatch(t)) {
      await Store.I.clearNotes();
      return const SkillResult('Cleared your notes.');
    }
    if (RegExp(r'\b(?:read|list|show|what\s+are)\b').hasMatch(t)) {
      final notes = Store.I.notes();
      if (notes.isEmpty) return const SkillResult('You have no notes.');
      final head = notes.take(5).toList();
      final body = head.asMap().entries.map((e) => '${e.key + 1}. ${e.value}').join('. ');
      final extra = notes.length > 5 ? ' And ${notes.length - 5} more.' : '';
      return SkillResult('You have ${notes.length} notes. $body.$extra');
    }
    final body = (m.groupNames.contains('body') ? m.namedGroup('body') : null) ??
        (m.groupNames.contains('body2') ? m.namedGroup('body2') : null) ??
        '';
    final note = body.trim();
    if (note.isEmpty) return const SkillResult('What should the note say?', success: false);
    await Store.I.addNote(note);
    return SkillResult('Noted: $note.');
  }
}
