import 'dart:async';

import '../core/skill.dart';

/// Timers, countdowns and short reminders. Fires via an in-app [Timer] and a
/// spoken announcement while the app is running (a phone-native background
/// alarm would need the notifications plugin — noted in the README).
class TimerSkill extends Skill {
  TimerSkill(this.announce);

  /// Called when a timer elapses (the UI speaks it).
  final void Function(String spoken) announce;

  final List<Timer> _active = [];

  @override
  String get name => 'timer';

  @override
  List<RegExp> get patterns => [
        RegExp(r'\b(?:set|start)\s+(?:a\s+)?(?:timer|alarm|countdown)\b'),
        RegExp(r'\bremind\s+me\b'),
        RegExp(r'\btimer\s+for\b'),
        RegExp(r'\bwake\s+me\s+(?:up\s+)?in\b'),
        RegExp(r'\b(?:ping|buzz|nudge|alert)\s+me\s+in\b'),
        RegExp(r'\bcancel\s+(?:all\s+)?(?:timers?|reminders?|alarms?)\b'),
      ];

  @override
  Future<SkillResult> run(SkillRequest request) async {
    final t = request.text;
    if (RegExp(r'\bcancel\b').hasMatch(t)) {
      for (final timer in _active) {
        timer.cancel();
      }
      final n = _active.length;
      _active.clear();
      return SkillResult(n == 0 ? 'No timers were running.' : 'Cancelled $n timer${n == 1 ? '' : 's'}.');
    }

    final secs = _parseDuration(t);
    if (secs <= 0) {
      return const SkillResult('How long should the timer be?', success: false);
    }
    final what = _extractLabel(t);
    late Timer timer;
    timer = Timer(Duration(seconds: secs), () {
      _active.remove(timer);
      announce(what.isEmpty ? "Time's up." : "Reminder: $what.");
    });
    _active.add(timer);
    return SkillResult(
        'Okay, ${what.isEmpty ? 'timer' : 'reminder'} set for ${_spokenDuration(secs)}'
        '${what.isEmpty ? '' : ' to $what'}.');
  }

  static const _units = {
    'second': 1, 'seconds': 1, 'sec': 1, 'secs': 1,
    'minute': 60, 'minutes': 60, 'min': 60, 'mins': 60,
    'hour': 3600, 'hours': 3600, 'hr': 3600, 'hrs': 3600,
  };
  static const _words = {
    'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'fifteen': 15,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'forty five': 45, 'sixty': 60,
  };

  int _parseDuration(String text) {
    var t = text;
    if (RegExp(r'\bhalf\s+(?:an\s+)?hour\b').hasMatch(t)) t = t.replaceAll(RegExp(r'\bhalf\s+(?:an\s+)?hour\b'), '30 minutes');
    _words.forEach((w, n) {
      t = t.replaceAll(RegExp('\\b$w\\s+(?=(?:second|minute|hour|sec|min|hr)s?\\b)'), '$n ');
    });
    var total = 0;
    for (final m in RegExp(r'(\d+)\s*([a-z]+)').allMatches(t)) {
      final n = int.tryParse(m.group(1)!) ?? 0;
      final u = _units[m.group(2)!];
      if (u != null) total += n * u;
    }
    return total;
  }

  String _extractLabel(String t) {
    final m = RegExp(r'\bto\s+(.+)$').firstMatch(t);
    if (m != null) return m.group(1)!.trim();
    return '';
  }

  String _spokenDuration(int s) {
    final parts = <String>[];
    final h = s ~/ 3600, m = (s % 3600) ~/ 60, sec = s % 60;
    if (h > 0) parts.add('$h hour${h == 1 ? '' : 's'}');
    if (m > 0) parts.add('$m minute${m == 1 ? '' : 's'}');
    if (sec > 0 && h == 0) parts.add('$sec second${sec == 1 ? '' : 's'}');
    return parts.join(' and ');
  }
}
