/// Server address editor with a live connection test, one-tap pairing
/// (LAN discovery + 6-digit code), plus the gesture-activation toggle and
/// sensitivity slider.
///
/// The address is the LAN `host:port` of the machine running
/// `python main.py --serve` (port 8710 by default, see config.yaml).
/// "Pair with MEDO" fills the address AND the auth token automatically:
/// MEDO is found by UDP broadcast, shows a 6-digit code on the PC screen,
/// and typing that code here proves you're at the machine.
library;

import 'package:flutter/material.dart';

import 'discovery.dart';
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
  final _codeController = TextEditingController();
  String? _status;
  Color _statusColor = Colors.white54;
  bool _testing = false;
  bool _pairing = false;       // discovery/pair-start in flight
  bool _awaitingCode = false;  // code flashed on the PC; waiting for input
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
    _codeController.dispose();
    super.dispose();
  }

  void _setStatus(String text, Color color) {
    setState(() {
      _status = text;
      _statusColor = color;
    });
  }

  /// One-tap pairing: discover MEDO on the LAN, have it flash a code on the
  /// PC, and swap that code for the token — no typing of IPs or tokens.
  Future<void> _pairWithMedo() async {
    setState(() {
      _pairing = true;
      _awaitingCode = false;
    });
    _setStatus('Looking for MEDO…', Colors.white54);
    try {
      final found = await MedoDiscovery.discover();
      if (found == null) {
        _setStatus("No MEDO found — same Wi-Fi as the PC?", Colors.orangeAccent);
        return;
      }
      _controller.text = found.address;
      await JarvisClient(found.address).pairStart();
      setState(() => _awaitingCode = true);
      _setStatus('Found ${found.name} ✓ — enter the code shown on the PC',
          Colors.cyanAccent);
    } on JarvisException catch (e) {
      _setStatus(e.message, Colors.orangeAccent);
    } finally {
      setState(() => _pairing = false);
    }
  }

  Future<void> _confirmPairCode() async {
    final code = _codeController.text.trim();
    if (code.length != 6) {
      _setStatus('The code has 6 digits.', Colors.orangeAccent);
      return;
    }
    setState(() => _pairing = true);
    try {
      final address = _controller.text.trim();
      final token = await JarvisClient(address).pairConfirm(code);
      _tokenController.text = token;
      await AppSettings.saveAddress(address);
      await AppSettings.saveToken(token);
      _codeController.clear();
      setState(() => _awaitingCode = false);
      final name = await JarvisClient(address, token).ping();
      _setStatus('Paired with $name ✓', Colors.greenAccent);
    } on JarvisException catch (e) {
      _setStatus(e.message, Colors.orangeAccent); // wrong code → try again
    } finally {
      setState(() => _pairing = false);
    }
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
                const SizedBox(height: 8),
                // The happy path: find MEDO + fetch the token, zero typing
                // (besides the 6-digit code MEDO shows on the PC screen).
                FilledButton.icon(
                  onPressed: _pairing ? null : _pairWithMedo,
                  icon: const Icon(Icons.wifi_tethering, size: 14),
                  label: const Text('Pair with MEDO',
                      style: TextStyle(fontSize: 12)),
                  style: FilledButton.styleFrom(
                    visualDensity: VisualDensity.compact,
                    backgroundColor: Colors.cyan.shade700,
                  ),
                ),
                if (_awaitingCode) ...[
                  const SizedBox(height: 8),
                  TextField(
                    controller: _codeController,
                    style: const TextStyle(
                        color: Colors.white, fontSize: 18, letterSpacing: 6),
                    keyboardType: TextInputType.number,
                    textAlign: TextAlign.center,
                    maxLength: 6,
                    decoration: const InputDecoration(
                      isDense: true,
                      counterText: '',
                      hintText: '••••••',
                      hintStyle: TextStyle(color: Colors.white24, fontSize: 18),
                      enabledBorder: UnderlineInputBorder(
                        borderSide: BorderSide(color: Colors.cyanAccent),
                      ),
                    ),
                    onSubmitted: (_) => _confirmPairCode(),
                  ),
                  const SizedBox(height: 6),
                  FilledButton(
                    onPressed: _pairing ? null : _confirmPairCode,
                    style: FilledButton.styleFrom(
                      visualDensity: VisualDensity.compact,
                      backgroundColor: Colors.cyan.shade800,
                    ),
                    child:
                        const Text('Confirm code', style: TextStyle(fontSize: 12)),
                  ),
                ],
                const SizedBox(height: 12),
                const Text(
                  'or set up manually:',
                  style: TextStyle(color: Colors.white24, fontSize: 9),
                ),
                const SizedBox(height: 6),
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
