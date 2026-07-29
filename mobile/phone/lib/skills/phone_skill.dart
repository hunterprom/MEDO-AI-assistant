import '../core/native.dart';
import '../core/skill.dart';

/// Native Android actions: flashlight, volume, open an app, call, text.
/// All go through the Kotlin MethodChannel; every one degrades to a spoken
/// "I couldn't…" instead of throwing.
class PhoneSkill extends Skill {
  @override
  String get name => 'phone';

  @override
  List<RegExp> get patterns => [
        RegExp(r'\b(?:turn\s+(?:on|off)\s+(?:the\s+)?)?(?:flashlight|torch|flash)\b'),
        RegExp(r'\b(?:set\s+)?volume\s+(?:to\s+)?(?<level>\d{1,3})\b'),
        RegExp(r'\b(?:turn\s+(?:it\s+)?(?<updown>up|down)|volume\s+(?<ud2>up|down)'
            r'|louder|quieter|too\s+loud|too\s+quiet|max\s+volume|mute)\b'),
        RegExp(r'\b(?:open|launch|start)\s+(?<app>[a-z0-9 .]+?)(?:\s+app)?\s*$'),
        RegExp(r'\b(?:call|phone|dial)\s+(?<num>[\d +().-]{3,})'),
        RegExp(r'\b(?:text|message|sms)\s+(?<smsnum>[\d +().-]{3,})'
            r'(?:\s+(?:saying|that)\s+(?<body>.+))?'),
      ];

  @override
  Future<SkillResult> run(SkillRequest request) async {
    final t = request.text;
    final m = request.match;

    // flashlight
    if (RegExp(r'\b(?:flashlight|torch|flash)\b').hasMatch(t)) {
      final off = t.contains('off');
      final ok = await Native.flashlight(!off);
      return ok
          ? SkillResult('Flashlight ${off ? 'off' : 'on'}.')
          : const SkillResult("I couldn't control the flashlight.", success: false);
    }

    // volume  (bare "turn up"/"turn down"/"turn it up" carry no 'volume' word
    // but pattern 3 matched them via the 'updown' group — enter here too)
    if (t.contains('volume') ||
        RegExp(r'\b(?:louder|quieter|mute|too\s+loud|too\s+quiet)\b').hasMatch(t) ||
        RegExp(r'\bturn\s+(?:it\s+)?(?:up|down)\b').hasMatch(t)) {
      final level = m.groupNames.contains('level') ? int.tryParse(m.namedGroup('level') ?? '') : null;
      if (t.contains('max')) {
        return _vol(await Native.volume(setPercent: 100), 'at max');
      }
      if (level != null) {
        return _vol(await Native.volume(setPercent: level), 'to $level percent');
      }
      final up = RegExp(r'\b(?:up|louder|too\s+quiet)\b').hasMatch(t);
      final down = RegExp(r'\b(?:down|quieter|too\s+loud)\b').hasMatch(t);
      if (t.contains('mute')) return _vol(await Native.volume(setPercent: 0), 'muted');
      if (up || down) return _vol(await Native.volume(step: up ? 1 : -1), up ? 'up' : 'down');
      return const SkillResult('Volume up, down, or to a number?', success: false);
    }

    // call / text
    final num = m.groupNames.contains('num') ? m.namedGroup('num') : null;
    if (num != null && num.trim().isNotEmpty) {
      final ok = await Native.dial(num.trim());
      return ok
          ? SkillResult('Opening the dialer for ${num.trim()}.')
          : const SkillResult("I couldn't open the dialer.", success: false);
    }
    final smsNum = m.groupNames.contains('smsnum') ? m.namedGroup('smsnum') : null;
    if (smsNum != null && smsNum.trim().isNotEmpty) {
      final body = (m.groupNames.contains('body') ? m.namedGroup('body') : null) ?? '';
      final ok = await Native.sms(smsNum.trim(), body.trim());
      return ok
          ? SkillResult('Opening a message to ${smsNum.trim()}.')
          : const SkillResult("I couldn't open the messenger.", success: false);
    }

    // open app
    final app = m.groupNames.contains('app') ? m.namedGroup('app')?.trim() : null;
    if (app != null && app.isNotEmpty) {
      final launched = await Native.openApp(app);
      return launched != null
          ? SkillResult('Opening $launched.')
          : SkillResult("I couldn't find an app called $app.", success: false);
    }

    return const SkillResult("I'm not sure what to do on the phone.", success: false);
  }

  SkillResult _vol(bool ok, String desc) => ok
      ? SkillResult('Volume $desc.')
      : const SkillResult("I couldn't change the volume.", success: false);
}
