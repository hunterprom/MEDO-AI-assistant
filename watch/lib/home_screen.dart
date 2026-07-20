/// Tap-to-talk screen: mic button → on-watch speech recognition → send the
/// transcript to Jarvis → show the reply and speak it aloud.
///
/// Designed for a small round display: everything important sits in the
/// center circle, one glanceable line of status on top, reply text in the
/// middle, controls at the bottom. A keyboard fallback covers emulators and
/// noisy rooms.
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_recognition_error.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';

import 'gesture_trigger.dart';
import 'ilioski_logo.dart';
import 'jarvis_client.dart';
import 'settings.dart';
import 'settings_screen.dart';

/// Mirrors Jarvis's own IDLE → LISTENING → THINKING → SPEAKING cycle.
enum Phase { idle, listening, thinking, speaking, error }

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  final SpeechToText _speech = SpeechToText();
  final FlutterTts _tts = FlutterTts();
  late final AnimationController _pulse;

  JarvisClient _client = JarvisClient(AppSettings.defaultAddress);
  late final GestureTrigger _gesture;
  bool _gestureEnabled = AppSettings.defaultGestureEnabled;
  bool _speechAvailable = false;
  Phase _phase = Phase.idle;
  String _display = '';
  String? _routeTag; // e.g. "FAST · datetime · 12 ms", shown under a reply

  /// Nothing asked yet — show the branded idle face instead of reply text.
  bool get _pristine => _display.isEmpty && _routeTag == null;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _gesture = GestureTrigger(onTrigger: _onGestureTrigger);
    _init();
  }

  Future<void> _init() async {
    final address = await AppSettings.loadAddress();
    _client = JarvisClient(address, await AppSettings.loadToken());

    _speechAvailable = await _speech.initialize(
      onError: _onSpeechError,
      onStatus: (_) {},
    );

    _tts.setCompletionHandler(() {
      if (mounted && _phase == Phase.speaking) {
        setState(() => _phase = Phase.idle);
      }
    });

    await _reloadGestureSettings();

    if (mounted && !_speechAvailable) {
      setState(() {
        _display = 'Voice input unavailable here — use the keyboard button.';
      });
    }
  }

  /// Pull the gesture toggle/threshold from storage and (re)arm the
  /// detector accordingly. Called on startup and after leaving settings.
  Future<void> _reloadGestureSettings() async {
    _gestureEnabled = await AppSettings.loadGestureEnabled();
    _gesture.threshold = await AppSettings.loadGestureThreshold();
    _syncGestureDetection();
  }

  /// Run flick detection only when it could act: feature on, speech works,
  /// and we're not already listening/thinking/speaking. Stopping the sensor
  /// (instead of just ignoring triggers) also saves watch battery.
  void _syncGestureDetection() {
    final shouldRun = _gestureEnabled &&
        _speechAvailable &&
        (_phase == Phase.idle || _phase == Phase.error);
    if (shouldRun) {
      _gesture.start();
    } else {
      _gesture.stop();
    }
  }

  /// Double wrist-flick detected: behave exactly like a mic-button tap,
  /// with a slightly stronger haptic so the wrist feels the confirmation.
  void _onGestureTrigger() {
    if (!mounted) return;
    if (_phase != Phase.idle && _phase != Phase.error) return;
    HapticFeedback.mediumImpact();
    _toggleListening();
  }

  @override
  void dispose() {
    _gesture.dispose();
    _pulse.dispose();
    _speech.cancel();
    _tts.stop();
    super.dispose();
  }

  void _setPhase(Phase phase) {
    setState(() => _phase = phase);
    if (phase == Phase.listening) {
      _pulse.repeat(reverse: true);
    } else {
      _pulse.stop();
      _pulse.value = 0;
    }
    _syncGestureDetection();
  }

  // --- voice flow ---

  Future<void> _toggleListening() async {
    if (_phase == Phase.listening) {
      await _speech.stop(); // a final result will still arrive
      return;
    }
    if (_phase == Phase.thinking) return; // request in flight
    await _tts.stop();
    HapticFeedback.lightImpact();
    setState(() {
      _display = '';
      _routeTag = null;
    });
    _setPhase(Phase.listening);
    await _speech.listen(
      onResult: _onSpeechResult,
      listenFor: const Duration(seconds: 15),
      pauseFor: const Duration(seconds: 2),
      listenOptions: SpeechListenOptions(
        partialResults: true,
        cancelOnError: true,
      ),
    );
  }

  void _onSpeechResult(SpeechRecognitionResult result) {
    final words = result.recognizedWords.trim();
    if (words.isNotEmpty) {
      setState(() => _display = words);
    }
    if (result.finalResult) {
      if (words.isEmpty) {
        setState(() => _display = "Didn't catch that — try again.");
        _setPhase(Phase.idle);
      } else {
        _send(words);
      }
    }
  }

  void _onSpeechError(SpeechRecognitionError error) {
    if (!mounted || _phase != Phase.listening) return;
    setState(() {
      _display = error.errorMsg == 'error_no_match'
          ? "Didn't catch that — try again."
          : 'Mic error: ${error.errorMsg}';
    });
    _setPhase(Phase.idle);
  }

  Future<void> _send(String text) async {
    setState(() {
      _display = text;
      _routeTag = null;
    });
    _setPhase(Phase.thinking);
    try {
      final reply = await _client.ask(text);
      if (!mounted) return;
      HapticFeedback.mediumImpact();
      setState(() {
        _display = reply.speech;
        _routeTag = reply.skill == null
            ? '${reply.path} · ${reply.latencyMs.round()} ms'
            : '${reply.path} · ${reply.skill} · ${reply.latencyMs.round()} ms';
      });
      _setPhase(Phase.speaking);
      await _tts.speak(reply.speech);
    } on JarvisException catch (e) {
      if (!mounted) return;
      setState(() => _display = e.message);
      _setPhase(Phase.error);
    }
  }

  // --- keyboard fallback (emulator, noisy rooms) ---

  Future<void> _typeInstead() async {
    final controller = TextEditingController();
    final text = await showDialog<String>(
      context: context,
      builder: (context) => Dialog(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: TextField(
            controller: controller,
            autofocus: true,
            textInputAction: TextInputAction.send,
            decoration: const InputDecoration(hintText: 'Ask Jarvis…'),
            onSubmitted: (value) => Navigator.pop(context, value),
          ),
        ),
      ),
    );
    final trimmed = text?.trim() ?? '';
    if (trimmed.isNotEmpty) await _send(trimmed);
  }

  Future<void> _openSettings() async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const SettingsScreen()),
    );
    _client = JarvisClient(
      await AppSettings.loadAddress(),
      await AppSettings.loadToken(),
    );
    await _reloadGestureSettings();
  }

  // --- UI ---

  static const _phaseColors = {
    Phase.idle: Colors.cyanAccent,
    Phase.listening: Colors.redAccent,
    Phase.thinking: Colors.amberAccent,
    Phase.speaking: Colors.greenAccent,
    Phase.error: Colors.orangeAccent,
  };

  static const _phaseLabels = {
    Phase.idle: 'JARVIS',
    Phase.listening: 'Listening…',
    Phase.thinking: 'Thinking…',
    Phase.speaking: 'Speaking',
    Phase.error: 'Offline',
  };

  @override
  Widget build(BuildContext context) {
    final color = _phaseColors[_phase]!;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Padding(
        // Generous inset keeps content inside the round face.
        padding: const EdgeInsets.symmetric(horizontal: 26, vertical: 18),
        child: Column(
          children: [
            Text(
              _phaseLabels[_phase]!,
              style: TextStyle(
                color: color,
                fontSize: 12,
                fontWeight: FontWeight.bold,
                letterSpacing: 2,
              ),
            ),
            const SizedBox(height: 6),
            Expanded(
              child: Center(
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 250),
                  child: _pristine ? _idleFace() : _replyView(),
                ),
              ),
            ),
            const SizedBox(height: 6),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _SmallButton(icon: Icons.keyboard, onTap: _typeInstead),
                const SizedBox(width: 14),
                _MicButton(
                  color: color,
                  listening: _phase == Phase.listening,
                  busy: _phase == Phase.thinking,
                  pulse: _pulse,
                  onTap: _speechAvailable ? _toggleListening : _typeInstead,
                ),
                const SizedBox(width: 14),
                _SmallButton(icon: Icons.settings, onTap: _openSettings),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _idleFace() {
    return Column(
      key: const ValueKey('idle'),
      mainAxisSize: MainAxisSize.min,
      children: [
        IlioskiLogo(size: 62, glow: _phase == Phase.speaking),
        const SizedBox(height: 8),
        const Text(
          'ILIOSKI',
          style: TextStyle(
            color: Colors.white38,
            fontSize: 9,
            letterSpacing: 4,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }

  Widget _replyView() {
    return SingleChildScrollView(
      key: ValueKey(_display),
      child: Column(
        children: [
          Text(
            _display,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white, fontSize: 14),
          ),
          if (_routeTag != null) ...[
            const SizedBox(height: 4),
            Text(
              _routeTag!,
              style: const TextStyle(color: Colors.white38, fontSize: 9),
            ),
          ],
        ],
      ),
    );
  }
}

class _MicButton extends StatelessWidget {
  const _MicButton({
    required this.color,
    required this.listening,
    required this.busy,
    required this.pulse,
    required this.onTap,
  });

  final Color color;
  final bool listening;
  final bool busy;
  final Animation<double> pulse;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: SizedBox(
        width: 68,
        height: 68,
        child: Stack(
          alignment: Alignment.center,
          children: [
            // Breathing halo while listening.
            AnimatedBuilder(
              animation: pulse,
              builder: (context, _) {
                if (!listening) return const SizedBox.shrink();
                final t = pulse.value;
                return Container(
                  width: 56 + 12 * t,
                  height: 56 + 12 * t,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: color.withOpacity(0.6 - 0.4 * t),
                      width: 2,
                    ),
                  ),
                );
              },
            ),
            AnimatedContainer(
              duration: const Duration(milliseconds: 250),
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: listening ? color : Colors.transparent,
                border: Border.all(color: color, width: 2),
              ),
              child: busy
                  ? Padding(
                      padding: const EdgeInsets.all(16),
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: color,
                      ),
                    )
                  : Icon(
                      listening ? Icons.stop : Icons.mic,
                      color: listening ? Colors.black : color,
                      size: 26,
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SmallButton extends StatelessWidget {
  const _SmallButton({required this.icon, required this.onTap});

  final IconData icon;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: onTap,
      icon: Icon(icon, color: Colors.white54, size: 18),
      visualDensity: VisualDensity.compact,
    );
  }
}
