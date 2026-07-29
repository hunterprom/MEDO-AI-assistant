/// Bridge to Android device features that have no pure-Dart API.
///
/// One MethodChannel to the Kotlin side (MainActivity) — no extra plugins.
/// Every call is best-effort: a missing capability returns a spoken-friendly
/// bool/string rather than throwing, so a skill can report it gracefully.
library;

import 'package:flutter/services.dart';

class Native {
  static const _ch = MethodChannel('medo/native');

  /// Turn the camera flashlight on/off. Returns true on success.
  static Future<bool> flashlight(bool on) async {
    try {
      return (await _ch.invokeMethod<bool>('flashlight', {'on': on})) ?? false;
    } catch (_) {
      return false;
    }
  }

  /// Nudge media volume up/down, or set an absolute 0-100 percent.
  static Future<bool> volume({int? setPercent, int step = 0}) async {
    try {
      return (await _ch.invokeMethod<bool>(
              'volume', {'set': setPercent, 'step': step})) ??
          false;
    } catch (_) {
      return false;
    }
  }

  /// Launch an installed app by a fuzzy label ("spotify", "camera"). Returns
  /// the launched app's label, or null if nothing matched.
  static Future<String?> openApp(String query) async {
    try {
      return await _ch.invokeMethod<String>('openApp', {'query': query});
    } catch (_) {
      return null;
    }
  }

  /// Open the dialer (does NOT auto-call) or the SMS composer.
  static Future<bool> dial(String number) async => _launch('dial', number);
  static Future<bool> sms(String number, String body) async {
    try {
      return (await _ch.invokeMethod<bool>('sms', {'number': number, 'body': body})) ??
          false;
    } catch (_) {
      return false;
    }
  }

  /// Open a web URL in the browser.
  static Future<bool> openUrl(String url) async => _launch('openUrl', url);

  /// Open the system camera and return the captured photo as a base64 JPEG
  /// (downscaled to ≤1024 px), or null if the user cancelled / it failed.
  static Future<String?> captureImage() async {
    try {
      return await _ch.invokeMethod<String>('captureImage');
    } catch (_) {
      return null;
    }
  }

  static Future<bool> _launch(String method, String arg) async {
    try {
      return (await _ch.invokeMethod<bool>(method, {'value': arg})) ?? false;
    } catch (_) {
      return false;
    }
  }
}
