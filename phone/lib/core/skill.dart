/// The on-device skill contract — a thin Dart mirror of MEDO's Python skills.
///
/// A skill declares regex `patterns`; the router picks the first whose pattern
/// matches and runs it. Everything runs on the phone; only skills that fetch
/// data (weather/search/news) or the cloud-chat fallback touch the network.
library;

/// One utterance to route, already lowercased/trimmed by the router.
class SkillRequest {
  const SkillRequest(this.text, this.match);

  /// The (lowercased) utterance.
  final String text;

  /// The regex match that selected this skill — capture groups carry args.
  final RegExpMatch match;
}

/// What a skill wants spoken back, and whether it succeeded.
class SkillResult {
  const SkillResult(this.speech, {this.success = true, this.data});

  final String speech;
  final bool success;

  /// Optional structured payload (e.g. a launched intent) for the UI/log.
  final Map<String, Object?>? data;
}

/// A deterministic capability. Pure where possible so it's testable.
abstract class Skill {
  /// Stable identifier, shown in the routing tag.
  String get name;

  /// Regexes that trigger this skill. First match in the list wins.
  List<RegExp> get patterns;

  /// Run the skill for a matched utterance.
  Future<SkillResult> run(SkillRequest request);

  /// Returns the first matching pattern's match, or null.
  RegExpMatch? matchOf(String text) {
    for (final p in patterns) {
      final m = p.firstMatch(text);
      if (m != null) return m;
    }
    return null;
  }
}
