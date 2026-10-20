/// Hands-free activation: detect a DOUBLE WRIST-FLICK on the watch.
///
/// Why a wrist flick and not a finger pinch? Wear OS keeps the OS-level
/// "Universal gestures" (pinch, double-pinch, …) inside its accessibility
/// service and does not forward those events to third-party apps, so a
/// Flutter app simply cannot observe a true pinch. What every app *can*
/// read is the raw accelerometer, and a deliberate flick of the wrist
/// produces a very distinctive spike on it — that is what we detect here.
///
/// Detection model (the math is also commented inline in [handleSample]):
///
/// 1. Read the *user* accelerometer (linear acceleration — the OS has
///    already subtracted gravity), so an arm held still reads ~0 m/s²
///    regardless of watch orientation.
/// 2. Reduce each 3-axis sample to its magnitude |a| = √(x² + y² + z²),
///    making detection orientation-independent.
/// 3. A "flick" is a rising edge where |a| crosses [threshold] (default
///    14 m/s² ≈ 1.4 g of gravity-free jerk — walking, typing and reaching
///    for a cup all stay well below that). Hysteresis: |a| must fall back
///    under 0.6 × threshold before another flick can be counted, so one
///    long swing is never double-counted.
/// 4. TWO flicks within [flickWindow] (700 ms), at least [minSpikeGap]
///    apart, fire [onTrigger] once, followed by a [cooldown] (2 s) during
///    which all samples are ignored — otherwise the arm still oscillating
///    after the gesture would re-trigger immediately.
library;

import 'dart:async';
import 'dart:math' as math;

import 'package:sensors_plus/sensors_plus.dart';

/// Detects a double wrist-flick on the user accelerometer and fires
/// [onTrigger]. Create it once, [start]/[stop] it around the phases where
/// hands-free activation makes sense, and [dispose] it with the widget.
class GestureTrigger {
  GestureTrigger({
    required this.onTrigger,
    this.threshold = defaultThreshold,
    this.flickWindow = const Duration(milliseconds: 700),
    this.cooldown = const Duration(seconds: 2),
    this.minSpikeGap = const Duration(milliseconds: 120),
  });

  /// Default spike threshold in m/s² (~1.4 g of gravity-free jerk).
  static const double defaultThreshold = 14.0;

  /// Called when a double flick is detected.
  final void Function() onTrigger;

  /// Acceleration-magnitude threshold in m/s² a spike must exceed.
  /// Mutable so the settings slider can retune a live detector.
  double threshold;

  /// Max time between the two flicks of a double-flick.
  final Duration flickWindow;

  /// Dead time after firing, so arm oscillation cannot re-trigger.
  final Duration cooldown;

  /// Min time between the two flicks — debounces a single jerk whose
  /// magnitude wobbles across the threshold twice within a few samples.
  final Duration minSpikeGap;

  StreamSubscription<UserAccelerometerEvent>? _subscription;

  /// True while |a| is above [threshold]; cleared only once it drops under
  /// the hysteresis floor (0.6 × threshold). Yields one spike per swing.
  bool _above = false;

  /// When the first flick of a candidate pair was seen, if any.
  DateTime? _firstSpikeAt;

  /// When [onTrigger] last fired; start of the cooldown window.
  DateTime? _lastFiredAt;

  /// Whether the accelerometer subscription is currently active.
  bool get isRunning => _subscription != null;

  /// Begin listening to the user accelerometer. No-op if already running.
  void start() {
    if (_subscription != null) return;
    _resetDetection();
    // 20 ms sampling (gameInterval) — fast enough to catch a ~100 ms flick
    // spike, cheap enough for a watch battery.
    _subscription = userAccelerometerEventStream(
      samplingPeriod: SensorInterval.gameInterval,
    ).listen(
      (UserAccelerometerEvent event) {
        // Orientation-independent magnitude: |a| = √(x² + y² + z²).
        final double magnitude = math.sqrt(
          event.x * event.x + event.y * event.y + event.z * event.z,
        );
        handleSample(magnitude, DateTime.now());
      },
      // No accelerometer (emulator) — silently degrade to tap-to-talk.
      onError: (Object _) => stop(),
      cancelOnError: true,
    );
  }

  /// Stop listening (e.g. while the app is already listening/speaking).
  void stop() {
    _subscription?.cancel();
    _subscription = null;
    _resetDetection();
  }

  /// Release resources. The trigger must not be reused afterwards.
  void dispose() => stop();

  /// Feed one magnitude sample through the detector state machine.
  ///
  /// Public (rather than private) so unit tests can drive the exact same
  /// code path with synthetic samples instead of a real accelerometer.
  void handleSample(double magnitude, DateTime now) {
    // --- cooldown gate ----------------------------------------------------
    final DateTime? fired = _lastFiredAt;
    if (fired != null && now.difference(fired) < cooldown) {
      // Inside the post-trigger dead time: keep the hysteresis flag honest
      // so we do not exit cooldown "mid-spike", but count nothing.
      _updateHysteresis(magnitude);
      _firstSpikeAt = null;
      return;
    }

    // --- rising-edge spike detection with hysteresis ----------------------
    // A spike is the *rising edge* of |a| crossing `threshold`. `_above`
    // stays set until |a| < 0.6 × threshold, so a swing hovering around the
    // threshold produces exactly one spike, not many.
    bool spike = false;
    if (!_above && magnitude >= threshold) {
      _above = true;
      spike = true;
    } else {
      _updateHysteresis(magnitude);
    }
    if (!spike) return;

    // --- pairing: is this the 1st or the 2nd flick? ------------------------
    final DateTime? first = _firstSpikeAt;
    if (first != null) {
      final Duration gap = now.difference(first);
      if (gap >= minSpikeGap && gap <= flickWindow) {
        // Two clean flicks inside the window → fire and enter cooldown.
        _lastFiredAt = now;
        _firstSpikeAt = null;
        onTrigger();
        return;
      }
      if (gap < minSpikeGap) {
        // Same physical jerk wobbling over the threshold — ignore it and
        // keep waiting for a genuine second flick.
        return;
      }
      // Too slow: the first flick expired; this spike restarts the pair.
    }
    _firstSpikeAt = now;
  }

  /// Re-arm spike detection once |a| falls under the hysteresis floor.
  void _updateHysteresis(double magnitude) {
    if (_above && magnitude < threshold * 0.6) _above = false;
  }

  void _resetDetection() {
    _above = false;
    _firstSpikeAt = null;
  }
}
