/// The intent router: try each on-device skill in priority order; if none
/// matches, fall back to the configured cloud model (or say chat is off).
library;

import 'package:medo_link/medo_link.dart';

import '../skills/calc_skill.dart';
import '../skills/datetime_skill.dart';
import '../skills/memory_skill.dart';
import '../skills/news_skill.dart';
import '../skills/notes_skill.dart';
import '../skills/phone_skill.dart';
import '../skills/timer_skill.dart';
import '../skills/weather_skill.dart';
import '../skills/websearch_skill.dart';
import 'llm.dart';
import 'native.dart';
import 'settings.dart';
import 'skill.dart';

/// One routed answer, tagged with how it was handled (for the UI).
class RouteResult {
  RouteResult(this.speech, this.path, {this.skill, this.data});
  final String speech;
  final String path; // 'FAST' or 'CHAT'
  final String? skill;
  final Map<String, Object?>? data;
}

class MedoRouter {
  MedoRouter(void Function(String) announce)
      : _skills = [
          // Order matters: specific/deterministic before the broad web/chat.
          MemorySkill(), // recall/forget/remember — anchored, checked first
          NotesSkill(),
          TimerSkill(announce),
          DateTimeSkill(),
          WeatherSkill(),
          NewsSkill(),
          PhoneSkill(),
          CalcSkill(),
          WebSearchSkill(), // broad "what is X" — last of the fast path
        ];

  final List<Skill> _skills;
  final LlmClient _llm = LlmClient();
  final List<(String, String)> _history = [];

  // Phrases that mean "look through the camera and tell me what you see".
  static final _cameraIntent = RegExp(
    r'\b(what (do|can) you see|what am i (looking at|holding)|look at (this|that|the)'
    r'|take a (photo|picture|pic|snap)|use the camera|open the camera|scan this'
    r'|(describe|read) (what you see|this|the (photo|picture|image|scene|room|label))'
    r'|can you see (this|that|the)|what.?s in front of (you|the camera))\b',
  );

  Future<RouteResult> route(String raw) async {
    final text = raw.trim();
    if (text.isEmpty) return RouteResult('', 'FAST');

    // Camera: capture a photo on the phone and have the vision model describe
    // it, right in the chat. Handled before everything else — even when linked
    // to the desktop, the camera in your hand is the phone's.
    if (_cameraIntent.hasMatch(text.toLowerCase())) {
      return _see(text);
    }

    // Linked to the desktop MEDO? Hand the whole utterance to its companion
    // API — the exact same system the watch app uses — and let the PC route it
    // through its full brain + skills. On-device skills are bypassed in this
    // mode (that's the point of connecting).
    if (AppSettings.I.useMedo) {
      try {
        final reply = await MedoClient(
                AppSettings.I.medoAddress, AppSettings.I.medoToken)
            .ask(text);
        _remember(text, reply.speech);
        return RouteResult(reply.speech, reply.path, skill: reply.skill ?? 'MEDO');
      } on MedoException catch (e) {
        return RouteResult(e.message, 'LINK');
      }
    }

    final lower = text.toLowerCase();

    for (final skill in _skills) {
      final m = skill.matchOf(lower);
      if (m == null) continue;
      final result = await skill.run(SkillRequest(lower, m));
      _remember(text, result.speech);
      return RouteResult(result.speech, 'FAST',
          skill: skill.name, data: result.data);
    }

    // No local skill matched — use the configured fallback backend.
    if (AppSettings.I.chatBackend == 'none') {
      return RouteResult(
          "I can't answer that on-device. Pick an AI model in settings.", 'FAST');
    }
    // Call the selected AI model's API and return its answer inline.
    try {
      final reply = await _llm.chat(text, _history);
      _remember(text, reply);
      return RouteResult(reply, 'CHAT',
          skill: AppSettings.providers[AppSettings.I.chatProvider]?.label ?? 'AI');
    } on LlmException catch (e) {
      return RouteResult(e.message, 'CHAT');
    }
  }

  /// Capture a photo and describe it. Checks the model can see BEFORE opening
  /// the camera, so we don't snap a picture we can't use.
  Future<RouteResult> _see(String prompt) async {
    if (!AppSettings.I.canSee) {
      final label =
          AppSettings.providers[AppSettings.I.chatProvider]?.label ?? 'the model';
      final why = AppSettings.I.chatEnabled
          ? "$label can't see images — switch to ChatGPT, Claude, or Gemini in settings."
          : 'To see, pick an AI model with vision (ChatGPT, Claude, or Gemini) in settings.';
      return RouteResult(why, 'VISION');
    }
    final image = await Native.captureImage();
    if (image == null || image.isEmpty) {
      return RouteResult('No photo taken.', 'VISION');
    }
    try {
      final desc = await _llm.see(prompt, image);
      _remember(prompt, desc);
      return RouteResult(desc, 'VISION',
          skill: AppSettings.providers[AppSettings.I.chatProvider]?.label ?? 'AI');
    } on LlmException catch (e) {
      return RouteResult(e.message, 'VISION');
    }
  }

  void _remember(String user, String reply) {
    _history.add((user, reply));
    if (_history.length > 6) _history.removeAt(0);
  }
}
