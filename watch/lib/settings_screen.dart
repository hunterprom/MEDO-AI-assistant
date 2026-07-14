/// Server address editor with a live connection test, plus the
/// gesture-activation toggle and sensitivity slider.
///
/// The address is the LAN `host:port` of the machine running
/// `python main.py --serve` (port 8710 by default, see config.yaml).
library;

import 'package:flutter/material.dart';

import 'jarvis_client.dart';
import 'settings.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _controller = TextEditingController();
  final _tokenController = TextEditingController();
  String? _status;
  Color _statusColor = Colors.white54;
  bool _testing = false;
  bool _gestureEnabled = AppSettings.defaultGestureEnabled;
  double _gestureThreshold = AppSettings.defaultGestureThreshold;

  @override
  void initState() {
    super.initState();
    AppSettings.loadAddress().then((address) {
      if (mounted) _controller.text = address;
    });
    AppSettings.loadToken().then((token) {
      if (mounted) _tokenController.text = token;
    });
    AppSettings.loadGestureEnabled().then((enabled) {
      if (mounted) setState(() => _gestureEnabled = enabled);
    });
    AppSettings.loadGestureThreshold().then((threshold) {
      if (mounted) setState(() => _gestureThreshold = threshold);
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _tokenController.dispose();
    super.dispose();
  }

  Future<void> _saveAndTest() async {
    final address = _controller.text.trim();
    if (address.isEmpty) return;
    final token = _tokenController.text.trim();
    await AppSettings.saveAddress(address);
    await AppSettings.saveToken(token);
    setState(() {
      _testing = true;
      _status = 'Connecting…';
      _statusColor = Colors.white54;
    });
    try {
      final name = await JarvisClient(address, token).ping();
      setState(() {
        _status = 'Connected to $name ✓';
        _statusColor = Colors.greenAccent;
      });
    } on JarvisException catch (e) {
      setState(() {
        _status = e.message;
        _statusColor = Colors.orangeAccent;
      });
    } finally {
      setState(() => _testing = false);
    }
  }

  /// Persist the gesture toggle immediately — no separate save step, same
  /// fire-and-forget style as the address field's "Save & test".
  Future<void> _setGestureEnabled(bool enabled) async {
    setState(() => _gestureEnabled = enabled);
    await AppSettings.saveGestureEnabled(enabled);
  }

  /// Persist the flick threshold once the slider is released.
  Future<void> _setGestureThreshold(double threshold) async {
    setState(() => _gestureThreshold = threshold);
    await AppSettings.saveGestureThreshold(threshold);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 20),
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'JARVIS SERVER',
                  style: TextStyle(
                    color: Colors.cyanAccent,
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 2,
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: _controller,
                  style: const TextStyle(color: Colors.white, fontSize: 13),
                  keyboardType: TextInputType.url,
                  decoration: const InputDecoration(
                    isDense: true,
                    hintText: AppSettings.defaultAddress,
                    hintStyle: TextStyle(color: Colors.white24, fontSize: 13),
                    enabledBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: Colors.white24),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                // Auth token from the server's secrets.local.yaml
                // (remote.token). Leave empty if server auth is disabled.
                TextField(
                  controller: _tokenController,
                  style: const TextStyle(color: Colors.white, fontSize: 13),
                  obscureText: true,
                  decoration: const InputDecoration(
                    isDense: true,
                    hintText: 'auth token (remote.token)',
                    hintStyle: TextStyle(color: Colors.white24, fontSize: 13),
                    enabledBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: Colors.white24),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                FilledButton(
                  onPressed: _testing ? null : _saveAndTest,
                  style: FilledButton.styleFrom(
                    visualDensity: VisualDensity.compact,
                    backgroundColor: Colors.cyan.shade800,
                  ),
                  child: const Text('Save & test', style: TextStyle(fontSize: 12)),
                ),
                if (_status != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    _status!,
                    textAlign: TextAlign.center,
                    style: TextStyle(color: _statusColor, fontSize: 11),
                  ),
                ],
                const SizedBox(height: 8),
                const Text(
                  'Run: python main.py --serve',
                  style: TextStyle(color: Colors.white24, fontSize: 9),
                ),
                const SizedBox(height: 18),
                const Text(
                  'GESTURE ACTIVATION',
                  style: TextStyle(
                    color: Colors.cyanAccent,
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 2,
                  ),
                ),
                const SizedBox(height: 4),
                SwitchListTile(
                  value: _gestureEnabled,
                  onChanged: _setGestureEnabled,
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  visualDensity: VisualDensity.compact,
                  activeColor: Colors.cyanAccent,
                  title: const Text(
                    'Double wrist-flick to talk',
                    style: TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                ),
                if (_gestureEnabled) ...[
                  Text(
                    'Sensitivity — flick force '
                    '${_gestureThreshold.toStringAsFixed(0)} m/s²',
                    style: const TextStyle(
                      color: Colors.white54,
                      fontSize: 10,
                    ),
                  ),
                  Slider(
                    value: _gestureThreshold,
                    min: AppSettings.minGestureThreshold,
                    max: AppSettings.maxGestureThreshold,
                    divisions: (AppSettings.maxGestureThreshold -
                            AppSettings.minGestureThreshold)
                        .round(),
                    activeColor: Colors.cyanAccent,
                    onChanged: (value) =>
                        setState(() => _gestureThreshold = value),
                    onChangeEnd: _setGestureThreshold,
                  ),
                  const Text(
                    'Lower = easier to trigger, higher = needs a firm flick',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white24, fontSize: 9),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
