/// On-device speech in (speech_to_text) and speech out (flutter_tts).
///
/// Both run on the phone using Android's own engines — no audio leaves the
/// device for recognition or synthesis.
library;

import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart';

/// Wraps the recognizer; expose a simple listen-once API for tap-to-talk.
class Ears {
  final SpeechToText _stt = SpeechToText();
  bool _available = false;

  Future<bool> init() async {
    _available = await _stt.initialize(onError: (_) {}, onStatus: (_) {});
    return _available;
  }

  bool get available => _available;
  bool get isListening => _stt.isListening;

  /// Listen for one utterance. [onPartial] streams interim text; the returned
  /// future completes with the final transcript (or '' if nothing was heard).
  Future<String> listenOnce({
    required void Function(String partial) onPartial,
    Duration listenFor = const Duration(seconds: 12),
  }) async {
    if (!_available) return '';
    final done = _Completer<String>();
    var last = '';
    await _stt.listen(
      onResult: (r) {
        last = r.recognizedWords;
        onPartial(last);
        if (r.finalResult && !done.isDone) done.complete(last.trim());
      },
      listenFor: listenFor,
      pauseFor: const Duration(seconds: 3),
      listenOptions: SpeechListenOptions(
        partialResults: true,
        cancelOnError: true,
      ),
    );
    // If the engine stops without a final result, resolve with what we heard.
    _stt.statusListener = (status) {
      if ((status == 'done' || status == 'notListening') && !done.isDone) {
        done.complete(last.trim());
      }
    };
    return done.future;
  }

  Future<void> stop() => _stt.stop();
}

/// Wraps flutter_tts; speaks and resolves when playback finishes.
class Mouth {
  final FlutterTts _tts = FlutterTts();
  bool _ready = false;

  Future<void> init() async {
    if (_ready) return;
    await _tts.awaitSpeakCompletion(true);
    await _tts.setSpeechRate(0.5);
    _ready = true;
  }

  Future<void> speak(String text) async {
    if (text.trim().isEmpty) return;
    await init();
    await _tts.stop();
    await _tts.speak(text);
  }

  Future<void> stop() => _tts.stop();
}

/// A tiny single-fire completer that tolerates a double-complete.
class _Completer<T> {
  T? _value;
  bool _done = false;
  final List<void Function(T)> _cbs = [];

  bool get isDone => _done;

  void complete(T value) {
    if (_done) return;
    _done = true;
    _value = value;
    for (final cb in _cbs) {
      cb(value);
    }
  }

  Future<T> get future async {
    if (_done) return _value as T;
    return Future<T>(() async {
      while (!_done) {
        await Future<void>.delayed(const Duration(milliseconds: 50));
      }
      return _value as T;
    });
  }
}
