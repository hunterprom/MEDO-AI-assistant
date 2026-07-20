/// Persisted app settings: server address + gesture activation.
library;

import 'package:shared_preferences/shared_preferences.dart';

class AppSettings {
  static const _addressKey = 'server_address';
  static const _tokenKey = 'server_token';
  static const _gestureEnabledKey = 'gesture_enabled';
  static const _gestureThresholdKey = 'gesture_threshold';

  /// Matej's Mac on the home LAN, so the watch works out of the box.
  /// If the Mac's DHCP lease ever changes this, fix it in the settings
  /// screen (gear icon). On the emulator use 10.0.2.2:8710 (= host machine).
  static const defaultAddress = '10.10.20.108:8710';

  /// Double wrist-flick activation is on by default — it costs one low-rate
  /// sensor subscription while idle and can always be switched off.
  static const defaultGestureEnabled = true;

  /// Flick threshold defaults/bounds in m/s² (see gesture_trigger.dart for
  /// the physics). Lower = more sensitive, higher = needs a harder flick.
  static const defaultGestureThreshold = 14.0;
  static const minGestureThreshold = 8.0;
  static const maxGestureThreshold = 25.0;

  static Future<String> loadAddress() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_addressKey) ?? defaultAddress;
  }

  static Future<void> saveAddress(String address) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_addressKey, address.trim());
  }

  /// Companion-API bearer token — on the PC it's in secrets.local.yaml
  /// under `remote.token` (generated on the first `--serve` run). Empty
  /// means "send none", which only works against pre-auth servers or with
  /// `remote.auth_enabled: false`.
  static Future<String> loadToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_tokenKey) ?? '';
  }

  static Future<void> saveToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, token.trim());
  }

  /// Whether the double wrist-flick hands-free trigger is enabled.
  static Future<bool> loadGestureEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_gestureEnabledKey) ?? defaultGestureEnabled;
  }

  static Future<void> saveGestureEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_gestureEnabledKey, enabled);
  }

  /// Flick threshold in m/s², clamped to the slider range so a corrupt
  /// preference can never make the trigger fire constantly or never.
  static Future<double> loadGestureThreshold() async {
    final prefs = await SharedPreferences.getInstance();
    final value = prefs.getDouble(_gestureThresholdKey) ??
        defaultGestureThreshold;
    return value.clamp(minGestureThreshold, maxGestureThreshold);
  }

  static Future<void> saveGestureThreshold(double threshold) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(
      _gestureThresholdKey,
      threshold.clamp(minGestureThreshold, maxGestureThreshold),
    );
  }
}
