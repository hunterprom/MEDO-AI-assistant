import '../core/skill.dart';

/// Calculator + simple unit/currency conversion — pure Dart, no network for
/// math; currency uses a small built-in rate table (offline, approximate).
class CalcSkill extends Skill {
  @override
  String get name => 'calc';

  @override
  List<RegExp> get patterns => [
        RegExp(r'\b(?:what(?:'
            r"'?s| is)?|calculate|compute)\s+.*[\d)]\s*[-+xX*/÷]\s*[\d(]"),
        RegExp(r'^\s*[-+]?[\d.]+\s*[-+xX*/÷].+'),
        RegExp(r'\bconvert\s+[\d.]+'),
        RegExp(r'\b[\d.]+\s*(?:km|kilometers?|miles?|kg|kilograms?|pounds?|lbs?|'
            r'celsius|fahrenheit|meters?|feet|ft|cm|inch(?:es)?)\b\s+(?:to|in)\b'),
        RegExp(r'\b[\d.]+\s*(?:usd|dollars?|eur|euros?|mkd|denars?|gbp|pounds?)\b'
            r'\s+(?:to|in)\b'),
      ];

  @override
  Future<SkillResult> run(SkillRequest request) async {
    final t = request.text;
    final conv = _convert(t);
    if (conv != null) return SkillResult(conv);
    final expr = _extractExpression(t);
    final value = expr == null ? null : _eval(expr);
    if (value == null) {
      return const SkillResult("I couldn't work that out.", success: false);
    }
    return SkillResult('That is ${_fmt(value)}.');
  }

  // --- arithmetic ----------------------------------------------------------
  String? _extractExpression(String t) {
    final m = RegExp(r'[-+]?[\d.]+(?:\s*[-+xX*/÷]\s*[-+]?[\d.]+)+').firstMatch(t);
    return m?.group(0);
  }

  /// A tiny left-to-right evaluator with */ precedence. Handles + - x * / ÷.
  double? _eval(String expr) {
    var s = expr.replaceAll('x', '*').replaceAll('X', '*').replaceAll('÷', '/').trim();
    // Peel a genuine leading unary sign first; otherwise the number token below
    // would greedily absorb a binary + / - ("2+2" -> ["2","+2"] -> crash).
    var lead = 1.0;
    if (s.startsWith('-')) {
      lead = -1.0;
      s = s.substring(1);
    } else if (s.startsWith('+')) {
      s = s.substring(1);
    }
    final tokens = RegExp(r'\d*\.?\d+|[-+*/]').allMatches(s).map((m) => m.group(0)!).toList();
    if (tokens.isEmpty) return null;
    // pass 1: * and /
    final nums = <double>[];
    final ops = <String>[];
    try {
      nums.add(lead * double.parse(tokens.first));
      for (var i = 1; i < tokens.length; i += 2) {
        final op = tokens[i];
        final n = double.parse(tokens[i + 1]);
        if (op == '*' || op == '/') {
          nums[nums.length - 1] =
              op == '*' ? nums.last * n : (n == 0 ? double.nan : nums.last / n);
        } else {
          ops.add(op);
          nums.add(n);
        }
      }
      var acc = nums.first;
      for (var i = 0; i < ops.length; i++) {
        acc = ops[i] == '+' ? acc + nums[i + 1] : acc - nums[i + 1];
      }
      return acc.isNaN ? null : acc;
    } catch (_) {
      return null;
    }
  }

  // --- conversions ---------------------------------------------------------
  static const _lengthToM = {
    'km': 1000.0, 'kilometer': 1000.0, 'kilometers': 1000.0,
    'mile': 1609.34, 'miles': 1609.34, 'meter': 1.0, 'meters': 1.0,
    'cm': 0.01, 'feet': 0.3048, 'ft': 0.3048, 'foot': 0.3048,
    'inch': 0.0254, 'inches': 0.0254,
  };
  static const _massToKg = {
    'kg': 1.0, 'kilogram': 1.0, 'kilograms': 1.0,
    'pound': 0.453592, 'pounds': 0.453592, 'lb': 0.453592, 'lbs': 0.453592,
  };
  // Rough, offline snapshot — good enough for a spoken estimate.
  static const _usdRate = {
    'usd': 1.0, 'dollar': 1.0, 'dollars': 1.0,
    'eur': 1.08, 'euro': 1.08, 'euros': 1.08,
    'gbp': 1.27, 'pound': 1.27, 'pounds': 1.27,
    'mkd': 0.0177, 'denar': 0.0177, 'denars': 0.0177,
  };

  String? _convert(String t) {
    final m = RegExp(
            r'([\d.]+)\s*([a-z°]+)\s+(?:to|in)\s+([a-z°]+)')
        .firstMatch(t);
    if (m == null) return null;
    final v = double.tryParse(m.group(1)!);
    final from = m.group(2)!, to = m.group(3)!;
    if (v == null) return null;

    // temperature
    if ((from.startsWith('c') || from.contains('celsius')) &&
        (to.startsWith('f') || to.contains('fahren'))) {
      return '${_fmt(v * 9 / 5 + 32)} degrees Fahrenheit.';
    }
    if ((from.startsWith('f') || from.contains('fahren')) &&
        (to.startsWith('c') || to.contains('celsius'))) {
      return '${_fmt((v - 32) * 5 / 9)} degrees Celsius.';
    }
    // length
    if (_lengthToM.containsKey(from) && _lengthToM.containsKey(to)) {
      return '${_fmt(v * _lengthToM[from]! / _lengthToM[to]!)} $to.';
    }
    // mass
    if (_massToKg.containsKey(from) && _massToKg.containsKey(to)) {
      return '${_fmt(v * _massToKg[from]! / _massToKg[to]!)} $to.';
    }
    // currency (approximate, offline)
    if (_usdRate.containsKey(from) && _usdRate.containsKey(to)) {
      final usd = v * _usdRate[from]!;
      return 'about ${_fmt(usd / _usdRate[to]!)} $to (approximate).';
    }
    return null;
  }

  String _fmt(double v) {
    if (v == v.roundToDouble()) return v.toStringAsFixed(0);
    final r = double.parse(v.toStringAsFixed(2));
    return r == r.roundToDouble() ? r.toStringAsFixed(0) : r.toString();
  }
}
