import 'dart:convert';

import 'package:http/http.dart' as http;

import '../core/settings.dart';
import '../core/skill.dart';

/// Weather via the free, keyless Open-Meteo API. Geocodes the named city (or
/// the configured default) then reads the current conditions.
class WeatherSkill extends Skill {
  @override
  String get name => 'weather';

  @override
  List<RegExp> get patterns => [
        RegExp(r'\bweather\b(?:\s+like)?(?:\s+(?:in|for|at)\s+(?<city>[a-z .-]+))?'),
        RegExp(r'\bforecast\b(?:\s+(?:in|for)\s+(?<city2>[a-z .-]+))?'),
        RegExp(r'\b(?:how\s+(?:hot|cold)|temperature)\b'),
        RegExp(r'\b(?:will\s+it|is\s+it\s+going\s+to)\s+rain\b'),
        RegExp(r'\bdo\s+i\s+need\s+(?:a|an|my)\s+(?:jacket|coat|umbrella)\b'),
        RegExp(r"\bwhat(?:'?s| is)\s+it\s+like\s+outside\b"),
      ];

  static const _codes = {
    0: 'clear', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast',
    45: 'foggy', 48: 'foggy', 51: 'drizzly', 53: 'drizzly', 55: 'drizzly',
    61: 'rainy', 63: 'rainy', 65: 'heavy rain', 71: 'snowy', 73: 'snowy',
    75: 'heavy snow', 80: 'rain showers', 81: 'rain showers', 82: 'heavy showers',
    95: 'thunderstorms', 96: 'thunderstorms', 99: 'thunderstorms',
  };

  @override
  Future<SkillResult> run(SkillRequest request) async {
    final m = request.match;
    var named = (m.groupNames.contains('city') ? m.namedGroup('city') : null) ??
        (m.groupNames.contains('city2') ? m.namedGroup('city2') : null);
    // The greedy city group runs to end-of-string, so it swallows trailing time
    // words ("weather in london tomorrow" -> "london tomorrow"). Strip them, or
    // geocoding the whole phrase finds nothing.
    named = named
        ?.replaceFirst(
            RegExp(r'\s+(?:today|tomorrow|tonight|now|right\s+now|this\s+\w+'
                r'|next\s+\w+|please)\s*$'),
            '')
        .trim();
    final city = ((named != null && named.isNotEmpty) ? named : AppSettings.I.city).trim();
    try {
      final geo = await _get(
          'https://geocoding-api.open-meteo.com/v1/search?count=1&name=${Uri.encodeQueryComponent(city)}');
      final results = (geo?['results'] as List<Object?>?);
      if (results == null || results.isEmpty) {
        return SkillResult("I couldn't find $city.", success: false);
      }
      final loc = results.first as Map<String, Object?>;
      final lat = loc['latitude'], lon = loc['longitude'];
      final name = loc['name'] as String? ?? city;
      final wx = await _get(
          'https://api.open-meteo.com/v1/forecast?latitude=$lat&longitude=$lon&current=temperature_2m,weather_code');
      final cur = wx?['current'] as Map<String, Object?>?;
      if (cur == null) return const SkillResult('The weather service gave no data.', success: false);
      final temp = (cur['temperature_2m'] as num?)?.round();
      final desc = _codes[(cur['weather_code'] as num?)?.toInt() ?? -1] ?? 'unclear';
      return SkillResult("It's $temp degrees and $desc in $name.");
    } catch (_) {
      return const SkillResult(
          "I couldn't reach the weather service — are you online?", success: false);
    }
  }

  Future<Map<String, Object?>?> _get(String url) async {
    final r = await http.get(Uri.parse(url)).timeout(const Duration(seconds: 8));
    if (r.statusCode >= 400) return null;
    return jsonDecode(r.body) as Map<String, Object?>;
  }
}
