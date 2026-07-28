/// Local persistence for notes, remembered facts, and reminders.
///
/// Backed by shared_preferences (JSON blobs) — no server, no account. Small
/// enough that the whole list loads at once; each mutation rewrites its key.
library;

import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

class Reminder {
  Reminder(this.text, this.dueMs);
  final String text;
  final int dueMs; // epoch millis when it should fire

  Map<String, Object?> toJson() => {'text': text, 'due': dueMs};
  factory Reminder.fromJson(Map<String, Object?> j) =>
      Reminder(j['text'] as String? ?? '', (j['due'] as num?)?.toInt() ?? 0);
}

/// One process-wide store. Call [init] once at startup.
class Store {
  Store._(this._prefs);
  final SharedPreferences _prefs;

  static Store? _instance;
  static Store get I => _instance!;

  static Future<Store> init() async =>
      _instance ??= Store._(await SharedPreferences.getInstance());

  // --- notes ---------------------------------------------------------------
  static const _kNotes = 'medo_notes';

  List<String> notes() => _prefs.getStringList(_kNotes) ?? <String>[];

  Future<void> addNote(String note) async {
    final list = notes()..add(note);
    await _prefs.setStringList(_kNotes, list);
  }

  Future<void> clearNotes() async => _prefs.remove(_kNotes);

  // --- facts (remember / recall) -------------------------------------------
  static const _kFacts = 'medo_facts';

  List<String> facts() => _prefs.getStringList(_kFacts) ?? <String>[];

  /// Store a fact; de-duplicates case-insensitively. Returns false if known.
  Future<bool> addFact(String fact) async {
    final list = facts();
    if (list.any((f) => f.toLowerCase() == fact.toLowerCase())) return false;
    list.add(fact);
    await _prefs.setStringList(_kFacts, list);
    return true;
  }

  /// Facts whose text contains any word of [query] (naive relevance).
  List<String> recall(String query) {
    final words = query
        .toLowerCase()
        .split(RegExp(r'\W+'))
        .where((w) => w.length > 2)
        .toSet();
    if (words.isEmpty) return facts();
    return facts()
        .where((f) => words.any((w) => f.toLowerCase().contains(w)))
        .toList();
  }

  Future<int> forget(String query) async {
    final keep = <String>[];
    var removed = 0;
    for (final f in facts()) {
      if (f.toLowerCase().contains(query.toLowerCase())) {
        removed++;
      } else {
        keep.add(f);
      }
    }
    await _prefs.setStringList(_kFacts, keep);
    return removed;
  }

  // --- reminders -----------------------------------------------------------
  static const _kReminders = 'medo_reminders';

  List<Reminder> reminders() {
    final raw = _prefs.getStringList(_kReminders) ?? <String>[];
    return raw
        .map((s) => Reminder.fromJson(jsonDecode(s) as Map<String, Object?>))
        .toList();
  }

  Future<void> addReminder(Reminder r) async {
    final list = _prefs.getStringList(_kReminders) ?? <String>[];
    list.add(jsonEncode(r.toJson()));
    await _prefs.setStringList(_kReminders, list);
  }

  Future<void> setReminders(List<Reminder> rs) async => _prefs.setStringList(
      _kReminders, rs.map((r) => jsonEncode(r.toJson())).toList());
}
