import 'package:flutter/material.dart';

import '../core/router.dart';
import '../core/settings.dart';
import '../core/voice.dart';
import 'hud_theme.dart';
import 'mic_orb.dart';
import 'settings_screen.dart';

class _Turn {
  _Turn(this.text, this.fromUser, {this.tag});
  final String text;
  final bool fromUser;
  final String? tag;
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final Ears _ears = Ears();
  final Mouth _mouth = Mouth();
  late final MedoRouter _router;

  final List<_Turn> _turns = [];
  final ScrollController _scroll = ScrollController();
  String _status = 'TAP TO TALK';
  bool _busy = false;
  bool _listening = false;

  @override
  void initState() {
    super.initState();
    _router = MedoRouter(_announce);
    _ears.init().then((ok) {
      if (mounted && !ok) setState(() => _status = 'GRANT MIC PERMISSION');
    });
  }

  void _announce(String spoken) {
    if (!mounted) return;
    setState(() => _turns.add(_Turn(spoken, false, tag: 'TIMER')));
    _mouth.speak(spoken);
    _scrollDown();
  }

  Future<void> _tap() async {
    if (_busy) return;
    if (_listening) {
      await _ears.stop();
      return;
    }
    await _mouth.stop();
    setState(() {
      _listening = true;
      _status = 'LISTENING…';
    });
    final text = await _ears.listenOnce(
      onPartial: (p) => setState(() => _status = p.isEmpty ? 'LISTENING…' : p.toUpperCase()),
    );
    setState(() => _listening = false);
    if (text.trim().isEmpty) {
      setState(() => _status = 'TAP TO TALK');
      return;
    }
    await _handle(text.trim());
  }

  Future<void> _handle(String text) async {
    setState(() {
      _turns.add(_Turn(text, true));
      _busy = true;
      _status = 'THINKING…';
    });
    _scrollDown();
    final result = await _router.route(text);
    if (!mounted) return;
    setState(() {
      _turns.add(_Turn(result.speech, false,
          tag: '${result.path}${result.skill != null ? ' · ${result.skill}' : ''}'));
      _busy = false;
      _status = 'TAP TO TALK';
    });
    _scrollDown();
    await _mouth.speak(result.speech);
  }

  void _scrollDown() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(_scroll.position.maxScrollExtent,
            duration: const Duration(milliseconds: 250), curve: Curves.easeOut);
      }
    });
  }

  Future<void> _type() async {
    final controller = TextEditingController();
    final text = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Hud.bgPanel,
        title: Text('TYPE A COMMAND', style: Hud.label(color: Hud.bright, size: 13)),
        content: TextField(
          controller: controller,
          autofocus: true,
          style: const TextStyle(color: Hud.text),
          textInputAction: TextInputAction.send,
          onSubmitted: (v) => Navigator.pop(ctx, v),
          decoration: const InputDecoration(hintText: 'Ask MEDO…'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('CANCEL')),
          FilledButton(
              onPressed: () => Navigator.pop(ctx, controller.text),
              child: const Text('SEND')),
        ],
      ),
    );
    if (text != null && text.trim().isNotEmpty) await _handle(text.trim());
  }

  Future<void> _openSettings() async {
    await Navigator.push(
        context, MaterialPageRoute(builder: (_) => const SettingsScreen()));
    if (mounted) setState(() {}); // linked-state may have changed → swap views
  }

  /// The most recent spoken reply, for the simple (linked) view.
  String get _lastReply {
    for (var i = _turns.length - 1; i >= 0; i--) {
      if (!_turns[i].fromUser) return _turns[i].text;
    }
    return '';
  }

  @override
  Widget build(BuildContext context) {
    final s = AppSettings.I;
    return Scaffold(
      backgroundColor: Hud.bgDeep,
      body: Container(
        decoration: Hud.cosmic,
        child: SafeArea(
          // Linked to the desktop MEDO → the phone is a thin remote for the
          // "bigger" assistant, so it shows a simpler, single-answer view. On
          // its own it runs the full on-device HUD (same feel as the main app).
          child: s.useMedo ? _simpleView(s) : _fullView(s),
        ),
      ),
    );
  }

  // --- full on-device HUD --------------------------------------------------

  Widget _fullView(AppSettings s) {
    return Column(
      children: [
        _topBar('ON-DEVICE'),
        Expanded(child: _turns.isEmpty ? _emptyHint() : _log()),
        const SizedBox(height: 8),
        MicOrb(listening: _listening, thinking: _busy, onTap: _tap),
        const SizedBox(height: 14),
        Text(_status,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Hud.label(color: _listening ? Hud.bright : Hud.muted, size: 12)),
        const SizedBox(height: 18),
        Padding(
          padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _ghostButton(Icons.keyboard, _type),
              const SizedBox(width: 20),
              _ghostButton(Icons.camera_alt_outlined,
                  () { if (!_busy) _handle('what do you see'); }),
              const SizedBox(width: 20),
              _ghostButton(Icons.settings, _openSettings),
            ],
          ),
        ),
      ],
    );
  }

  // --- simple thin-client view (linked to the desktop MEDO) ----------------

  Widget _simpleView(AppSettings s) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 14, 12, 6),
          child: Row(
            children: [
              Text(s.assistantName.toUpperCase(),
                  style: Hud.label(color: Hud.text, size: 15, spacing: 4)),
              const SizedBox(width: 10),
              const Icon(Icons.link, size: 14, color: Hud.ok),
              const Spacer(),
              IconButton(
                onPressed: _openSettings,
                icon: const Icon(Icons.settings, color: Hud.muted, size: 20),
                splashRadius: 20,
              ),
            ],
          ),
        ),
        Expanded(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 30),
              child: Text(
                _lastReply.isEmpty ? 'CONNECTED TO MEDO' : _lastReply,
                textAlign: TextAlign.center,
                style: _lastReply.isEmpty
                    ? Hud.label(size: 12, spacing: 2)
                    : const TextStyle(color: Hud.text, fontSize: 20, height: 1.35),
              ),
            ),
          ),
        ),
        MicOrb(listening: _listening, thinking: _busy, onTap: _tap, size: 210),
        const SizedBox(height: 16),
        Text(_status,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Hud.label(color: _listening ? Hud.bright : Hud.muted, size: 12)),
        const SizedBox(height: 8),
        Text('LINKED · ${s.medoAddress}', style: Hud.label(size: 9, spacing: 1)),
        const SizedBox(height: 26),
      ],
    );
  }

  Widget _topBar(String mode) => Padding(
        padding: const EdgeInsets.fromLTRB(20, 14, 20, 6),
        child: Row(
          children: [
            Container(
              width: 9,
              height: 9,
              margin: const EdgeInsets.only(right: 10),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Hud.primary,
                boxShadow: [BoxShadow(color: Hud.primary.withOpacity(0.8), blurRadius: 8)],
              ),
            ),
            Text(AppSettings.I.assistantName.toUpperCase(),
                style: Hud.label(color: Hud.text, size: 15, spacing: 4)),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                border: Border.all(color: Hud.panelBorder),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(mode, style: Hud.label(size: 9)),
            ),
          ],
        ),
      );

  Widget _emptyHint() => Center(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Text(
            'TIME · TIMERS · NOTES · WEATHER · CAMERA\n'
            'SEARCH · NEWS · MATH · PHONE ACTIONS',
            textAlign: TextAlign.center,
            style: Hud.label(size: 11, spacing: 1.5),
          ),
        ),
      );

  Widget _log() => ListView.builder(
        controller: _scroll,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        itemCount: _turns.length,
        itemBuilder: (_, i) => _bubble(_turns[i]),
      );

  Widget _bubble(_Turn turn) {
    return Container(
      alignment: turn.fromUser ? Alignment.centerRight : Alignment.centerLeft,
      margin: const EdgeInsets.symmetric(vertical: 5),
      child: Column(
        crossAxisAlignment:
            turn.fromUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
        children: [
          Container(
            constraints:
                BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.82),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: turn.fromUser
                  ? Hud.primary.withOpacity(0.14)
                  : Hud.bgPanel.withOpacity(0.85),
              border: Border.all(color: Hud.panelBorder),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(turn.text, style: const TextStyle(color: Hud.text, height: 1.3)),
          ),
          if (turn.tag != null)
            Padding(
              padding: const EdgeInsets.only(top: 3, left: 4, right: 4),
              child: Text(turn.tag!, style: Hud.label(size: 9, spacing: 1)),
            ),
        ],
      ),
    );
  }

  Widget _ghostButton(IconData icon, VoidCallback onTap) => InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(30),
        child: Container(
          width: 52,
          height: 52,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: Hud.panelBorder),
            color: Hud.bgPanel.withOpacity(0.6),
          ),
          child: Icon(icon, color: Hud.bright, size: 22),
        ),
      );

  @override
  void dispose() {
    _scroll.dispose();
    _mouth.stop();
    super.dispose();
  }
}
