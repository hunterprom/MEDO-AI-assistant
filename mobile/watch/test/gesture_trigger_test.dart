/// Unit tests for the double wrist-flick detector.
///
/// Drives [GestureTrigger.handleSample] with synthetic magnitude samples so
/// no real accelerometer (or platform channel) is needed.
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:jarvis_watch/gesture_trigger.dart';

void main() {
  /// A fixed origin keeps the synthetic timeline deterministic.
  final t0 = DateTime(2026, 1, 1);

  /// Builds a trigger that counts its own firings.
  (GestureTrigger, List<DateTime>) makeTrigger({double threshold = 14.0}) {
    final firings = <DateTime>[];
    late GestureTrigger trigger;
    trigger = GestureTrigger(
      onTrigger: () => firings.add(t0),
      threshold: threshold,
    );
    return (trigger, firings);
  }

  /// One flick: rest → spike above threshold → back to rest.
  void flick(GestureTrigger trigger, DateTime at, {double peak = 20.0}) {
    trigger.handleSample(1.0, at.subtract(const Duration(milliseconds: 40)));
    trigger.handleSample(peak, at);
    trigger.handleSample(1.0, at.add(const Duration(milliseconds: 40)));
  }

  test('two flicks within 700 ms fire exactly once', () {
    final (trigger, firings) = makeTrigger();
    flick(trigger, t0);
    flick(trigger, t0.add(const Duration(milliseconds: 400)));
    expect(firings.length, 1);
  });

  test('a single flick never fires', () {
    final (trigger, firings) = makeTrigger();
    flick(trigger, t0);
    // Quiet arm afterwards.
    trigger.handleSample(0.5, t0.add(const Duration(seconds: 3)));
    expect(firings, isEmpty);
  });

  test('two flicks further apart than the window do not fire', () {
    final (trigger, firings) = makeTrigger();
    flick(trigger, t0);
    flick(trigger, t0.add(const Duration(milliseconds: 900)));
    expect(firings, isEmpty);
  });

  test('sub-threshold motion never fires', () {
    final (trigger, firings) = makeTrigger();
    for (int i = 0; i < 50; i++) {
      trigger.handleSample(10.0, t0.add(Duration(milliseconds: 20 * i)));
    }
    expect(firings, isEmpty);
  });

  test('one long swing hovering above threshold counts as one flick', () {
    final (trigger, firings) = makeTrigger();
    // Magnitude stays above the hysteresis floor the whole time, so the
    // rising edge is counted once even though many samples exceed 14.
    for (int i = 0; i < 20; i++) {
      trigger.handleSample(18.0, t0.add(Duration(milliseconds: 20 * i)));
    }
    trigger.handleSample(1.0, t0.add(const Duration(milliseconds: 500)));
    expect(firings, isEmpty);
  });

  test('cooldown suppresses re-trigger for 2 s, then re-arms', () {
    final (trigger, firings) = makeTrigger();
    // First double flick fires.
    flick(trigger, t0);
    flick(trigger, t0.add(const Duration(milliseconds: 300)));
    expect(firings.length, 1);
    // A double flick 1 s later is inside the cooldown — ignored.
    flick(trigger, t0.add(const Duration(milliseconds: 1300)));
    flick(trigger, t0.add(const Duration(milliseconds: 1600)));
    expect(firings.length, 1);
    // After the 2 s cooldown a new double flick fires again.
    flick(trigger, t0.add(const Duration(milliseconds: 3000)));
    flick(trigger, t0.add(const Duration(milliseconds: 3300)));
    expect(firings.length, 2);
  });

  test('threshold is tunable at runtime', () {
    final (trigger, firings) = makeTrigger(threshold: 25.0);
    // 20 m/s² peaks are below a 25 m/s² threshold.
    flick(trigger, t0);
    flick(trigger, t0.add(const Duration(milliseconds: 300)));
    expect(firings, isEmpty);
    // Loosen the threshold (as the settings slider does) and retry.
    trigger.threshold = 14.0;
    flick(trigger, t0.add(const Duration(seconds: 5)));
    flick(trigger, t0.add(const Duration(seconds: 5, milliseconds: 300)));
    expect(firings.length, 1);
  });

  test('start/stop lifecycle flags are consistent without a sensor', () {
    final (trigger, _) = makeTrigger();
    expect(trigger.isRunning, isFalse);
    trigger.stop(); // stop before start is a safe no-op
    expect(trigger.isRunning, isFalse);
    trigger.dispose();
    expect(trigger.isRunning, isFalse);
  });
}
