import 'package:flutter/material.dart';
import 'package:medo_link/medo_link.dart';

import '../core/settings.dart';
import 'hud_theme.dart';

/// Configure the AI model that answers open-ended questions (inline in the
/// chat), the link to the desktop MEDO, and the weather city — HUD-styled.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _apiKey;
  late final TextEditingController _model;
  late final TextEditingController _customBase;
  late final TextEditingController _city;

  String _backend = 'cloud'; // 'none' | 'cloud'
  String _provider = 'groq';

  @override
  void initState() {
    super.initState();
    final s = AppSettings.I;
    _backend = s.chatBackend;
    _provider = s.chatProvider;
    _apiKey = TextEditingController(text: s.providerKey(_provider));
    _model = TextEditingController(text: s.providerModel(_provider));
    _customBase = TextEditingController(text: s.customBase);
    _city = TextEditingController(text: s.city);
  }

  /// Persist whatever's typed for the currently-shown provider.
  void _stashCurrent() {
    final s = AppSettings.I;
    s.setProviderKey(_provider, _apiKey.text);
    s.setProviderModel(_provider, _model.text);
    if (_provider == 'custom') s.customBase = _customBase.text;
  }

  void _selectProvider(String id) {
    _stashCurrent();
    final s = AppSettings.I;
    setState(() {
      _provider = id;
      _apiKey.text = s.providerKey(id);
      _model.text = s.providerModel(id);
      _customBase.text = id == 'custom' ? s.customBase : '';
    });
  }

  void _save() {
    _stashCurrent();
    final s = AppSettings.I;
    s.chatBackend = _backend;
    s.chatProvider = _provider;
    s.city = _city.text;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      backgroundColor: Hud.bgPanel,
      content: Text('SAVED', style: Hud.label(color: Hud.ok, size: 12)),
    ));
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final s = AppSettings.I;
    return Scaffold(
      backgroundColor: Hud.bgDeep,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Hud.bright),
        title: Text('SETTINGS', style: Hud.label(color: Hud.text, size: 15, spacing: 3)),
      ),
      body: Container(
        decoration: Hud.cosmic,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(18, 8, 18, 40),
          children: [
            _sectionTitle('OPEN-ENDED QUESTIONS'),
            _hint('Skills always run on the phone. Anything no skill caught goes '
                'to your chosen AI model, and its answer comes back here in the chat.'),
            const SizedBox(height: 12),
            _backendSelector(),
            const SizedBox(height: 16),
            if (_backend == 'cloud') ..._modelPicker(),
            if (_backend == 'none')
              _hint('Fully on-device — open-ended questions get "add a model".'),
            const SizedBox(height: 26),
            _sectionTitle('CONNECT TO MEDO'),
            _hint('Link to the desktop MEDO over Wi-Fi (same pairing as the '
                'watch). When routed, every request runs on the PC\'s full brain.'),
            const SizedBox(height: 12),
            _medoPanel(s),
            const SizedBox(height: 26),
            _sectionTitle('WEATHER'),
            const SizedBox(height: 10),
            _field(_city, 'DEFAULT CITY'),
            const SizedBox(height: 30),
            SizedBox(
              height: 50,
              child: FilledButton(
                onPressed: _save,
                style: FilledButton.styleFrom(
                  backgroundColor: Hud.primary.withOpacity(0.18),
                  side: const BorderSide(color: Hud.primary),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                child: Text('SAVE', style: Hud.label(color: Hud.bright, size: 14, spacing: 3)),
              ),
            ),
            const SizedBox(height: 18),
            _hint('Notes, facts and settings never leave the phone. Only the '
                'chosen model and the web/weather/news lookups touch the network.'),
          ],
        ),
      ),
    );
  }

  // --- backend on/off ------------------------------------------------------

  Widget _backendSelector() {
    Widget seg(String id, String label, IconData icon) {
      final on = _backend == id;
      return Expanded(
        child: GestureDetector(
          onTap: () => setState(() => _backend = id),
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 3),
            padding: const EdgeInsets.symmetric(vertical: 12),
            decoration: BoxDecoration(
              color: on ? Hud.primary.withOpacity(0.16) : Hud.bgPanel.withOpacity(0.5),
              border: Border.all(color: on ? Hud.primary : Hud.panelBorder),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              children: [
                Icon(icon, color: on ? Hud.bright : Hud.muted, size: 20),
                const SizedBox(height: 6),
                Text(label,
                    style: Hud.label(color: on ? Hud.bright : Hud.muted, size: 10, spacing: 1)),
              ],
            ),
          ),
        ),
      );
    }

    return Row(
      children: [
        seg('none', 'ON-DEVICE', Icons.phone_android),
        seg('cloud', 'AI MODEL', Icons.auto_awesome),
      ],
    );
  }

  // --- model picker --------------------------------------------------------

  List<Widget> _modelPicker() {
    final selectedHasKey = AppSettings.I.providerKey(_provider).isNotEmpty ||
        _apiKey.text.trim().isNotEmpty;
    return [
      Text('MODEL', style: Hud.label(color: Hud.primary, size: 11, spacing: 2)),
      const SizedBox(height: 8),
      Wrap(
        spacing: 8,
        runSpacing: 8,
        children: AppSettings.providers.entries.map((e) {
          final on = e.key == _provider;
          final hasKey = AppSettings.I.providerKey(e.key).isNotEmpty;
          return GestureDetector(
            onTap: () => _selectProvider(e.key),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
              decoration: BoxDecoration(
                color: on ? Hud.primary.withOpacity(0.18) : Hud.bgPanel.withOpacity(0.6),
                border: Border.all(color: on ? Hud.primary : Hud.panelBorder),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (hasKey)
                    const Padding(
                      padding: EdgeInsets.only(right: 6),
                      child: Icon(Icons.check_circle, size: 13, color: Hud.ok),
                    ),
                  Text(e.value.label.toUpperCase(),
                      style: Hud.label(color: on ? Hud.bright : Hud.text, size: 11, spacing: 1)),
                ],
              ),
            ),
          );
        }).toList(),
      ),
      const SizedBox(height: 14),
      _hint(_provider == 'custom'
          ? 'Any OpenAI-compatible endpoint. Enter its base URL, model, and key.'
          : 'MEDO calls ${AppSettings.providers[_provider]!.label} directly with '
              'your API key and shows the reply here. (The installed app can\'t '
              'hand its answer back to another app — the API key is how.)'),
      const SizedBox(height: 10),
      if (_provider == 'custom') _field(_customBase, 'BASE URL'),
      _field(_apiKey, '${AppSettings.providers[_provider]!.label.toUpperCase()} API KEY',
          obscure: true),
      _field(_model, 'MODEL'),
      if (!selectedHasKey)
        Padding(
          padding: const EdgeInsets.only(top: 2),
          child: Text('No key yet — add one to enable this model.',
              style: Hud.label(color: Hud.danger, size: 10, spacing: 0.5)),
        ),
    ];
  }

  // --- connect to MEDO -----------------------------------------------------

  Widget _medoPanel(AppSettings s) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Hud.bgPanel.withOpacity(0.6),
        border: Border.all(color: Hud.panelBorder),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(s.linked ? Icons.link : Icons.link_off,
                  color: s.linked ? Hud.ok : Hud.muted, size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  s.linked ? s.medoAddress : 'NOT LINKED',
                  style: Hud.label(color: s.linked ? Hud.text : Hud.muted, size: 12),
                ),
              ),
            ],
          ),
          if (s.linked) ...[
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: Text('ROUTE EVERYTHING TO MEDO',
                      style: Hud.label(color: Hud.text, size: 11)),
                ),
                Switch(
                  value: s.useMedo,
                  activeColor: Hud.bright,
                  activeTrackColor: Hud.primary.withOpacity(0.4),
                  onChanged: (v) => setState(() => s.useMedo = v),
                ),
              ],
            ),
          ],
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              // Primary path: approve on the PC — no code to read and type.
              _smallBtn(s.linked ? 'RE-CONNECT' : 'CONNECT', _connectApprove),
              // Fallback: the classic 6-digit code, and manual address entry.
              _smallBtn('USE CODE', _discoverAndPair),
              _smallBtn('MANUAL', _manualEntry),
              if (s.linked) _smallBtn('UNLINK', _unlink, danger: true),
            ],
          ),
        ],
      ),
    );
  }

  /// Approve-on-PC: discover MEDO, ask to connect, and wait for the user to
  /// tap Approve in the desktop HUD — no code to read off the screen and type.
  Future<void> _connectApprove() async {
    _toast('Searching the network…');
    final server = await MedoDiscovery.discover();
    if (!mounted) return;
    if (server == null) {
      _toast('No MEDO found. Same Wi-Fi? Try MANUAL.', bad: true);
      return;
    }
    try {
      final token = await MedoClient(server.address).connectViaApproval(
        name: 'Phone',
        kind: 'phone',
        onWaiting: () {
          if (mounted) {
            _toast('Approve on the PC: MEDO HUD → CONFIG → Easy connect.');
          }
        },
      );
      AppSettings.I
        ..medoAddress = server.address
        ..medoToken = token
        ..useMedo = true;
      if (mounted) setState(() {});
      _toast('Linked to ${server.name}.', ok: true);
    } on MedoException catch (e) {
      if (mounted) _toast(e.message, bad: true);
    }
  }

  Future<void> _discoverAndPair() async {
    _toast('Searching the network…');
    final server = await MedoDiscovery.discover();
    if (!mounted) return;
    if (server == null) {
      _toast('No MEDO found. Same Wi-Fi? Try MANUAL.', bad: true);
      return;
    }
    final client = MedoClient(server.address);
    try {
      await client.pairStart();
    } on MedoException catch (e) {
      _toast(e.message, bad: true);
      return;
    }
    if (!mounted) return;
    final code = await _askCode('${server.name} at ${server.address}');
    if (code == null || code.trim().isEmpty) return;
    try {
      final token = await client.pairConfirm(code.trim());
      AppSettings.I
        ..medoAddress = server.address
        ..medoToken = token
        ..useMedo = true;
      if (mounted) setState(() {});
      _toast('Linked to ${server.name}.', ok: true);
    } on MedoException catch (e) {
      _toast(e.message, bad: true);
    }
  }

  Future<String?> _askCode(String subtitle) {
    final c = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Hud.bgPanel,
        title: Text('ENTER PAIRING CODE', style: Hud.label(color: Hud.bright, size: 13)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Read the 6-digit code on\n$subtitle',
                style: Hud.label(size: 10, spacing: 0.5)),
            const SizedBox(height: 12),
            TextField(
              controller: c,
              autofocus: true,
              keyboardType: TextInputType.number,
              style: const TextStyle(color: Hud.text, fontSize: 22, letterSpacing: 8),
              textAlign: TextAlign.center,
              maxLength: 6,
              onSubmitted: (v) => Navigator.pop(ctx, v),
              decoration: const InputDecoration(counterText: '', hintText: '••••••'),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('CANCEL')),
          FilledButton(onPressed: () => Navigator.pop(ctx, c.text), child: const Text('PAIR')),
        ],
      ),
    );
  }

  Future<void> _manualEntry() async {
    final s = AppSettings.I;
    final addr = TextEditingController(text: s.medoAddress);
    final tok = TextEditingController(text: s.medoToken);
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Hud.bgPanel,
        title: Text('MANUAL LINK', style: Hud.label(color: Hud.bright, size: 13)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: addr,
              style: const TextStyle(color: Hud.text),
              decoration: const InputDecoration(labelText: 'host:port (e.g. 192.168.1.20:8710)'),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: tok,
              style: const TextStyle(color: Hud.text),
              decoration: const InputDecoration(labelText: 'token (optional)'),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('CANCEL')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('LINK')),
        ],
      ),
    );
    if (ok == true && addr.text.trim().isNotEmpty) {
      s
        ..medoAddress = addr.text
        ..medoToken = tok.text
        ..useMedo = true;
      if (mounted) setState(() {});
      _toast('Linked to ${addr.text.trim()}.', ok: true);
    }
  }

  void _unlink() {
    final s = AppSettings.I;
    s
      ..medoAddress = ''
      ..medoToken = ''
      ..useMedo = false;
    setState(() {});
    _toast('Unlinked.');
  }

  // --- shared bits ---------------------------------------------------------

  Widget _smallBtn(String label, VoidCallback onTap, {bool danger = false}) {
    final c = danger ? Hud.danger : Hud.bright;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          border: Border.all(color: danger ? Hud.danger.withOpacity(0.6) : Hud.panelBorder),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Text(label, style: Hud.label(color: c, size: 10, spacing: 1)),
      ),
    );
  }

  Widget _sectionTitle(String t) =>
      Text(t, style: Hud.label(color: Hud.primary, size: 12, spacing: 3));

  Widget _hint(String t) => Padding(
        padding: const EdgeInsets.only(top: 6),
        child: Text(t, style: const TextStyle(color: Hud.muted, fontSize: 12, height: 1.35)),
      );

  Widget _field(TextEditingController c, String label, {bool obscure = false}) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: TextField(
          controller: c,
          obscureText: obscure,
          style: const TextStyle(color: Hud.text),
          decoration: InputDecoration(
            labelText: label,
            labelStyle: Hud.label(size: 11),
            isDense: true,
            enabledBorder: const OutlineInputBorder(
                borderSide: BorderSide(color: Hud.panelBorder)),
            focusedBorder: const OutlineInputBorder(
                borderSide: BorderSide(color: Hud.primary)),
          ),
        ),
      );

  void _toast(String msg, {bool ok = false, bool bad = false}) {
    if (!mounted) return;
    final color = bad ? Hud.danger : (ok ? Hud.ok : Hud.text);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      backgroundColor: Hud.bgPanel,
      content: Text(msg, style: Hud.label(color: color, size: 12, spacing: 0.5)),
    ));
  }

  @override
  void dispose() {
    _apiKey.dispose();
    _model.dispose();
    _customBase.dispose();
    _city.dispose();
    super.dispose();
  }
}
