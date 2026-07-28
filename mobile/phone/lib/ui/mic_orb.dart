import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'hud_theme.dart';

/// The arc-reactor mic button: concentric cyan rings around a glowing core,
/// with a pulsing halo while listening and a rotating tick ring while thinking.
class MicOrb extends StatefulWidget {
  const MicOrb({
    super.key,
    required this.listening,
    required this.thinking,
    required this.onTap,
    this.size = 190,
  });

  final bool listening;
  final bool thinking;
  final VoidCallback onTap;
  final double size;

  @override
  State<MicOrb> createState() => _MicOrbState();
}

class _MicOrbState extends State<MicOrb> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(seconds: 4))
        ..repeat();

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: widget.onTap,
      child: AnimatedBuilder(
        animation: _c,
        builder: (_, __) => CustomPaint(
          size: Size.square(widget.size),
          painter: _OrbPainter(
            t: _c.value,
            listening: widget.listening,
            thinking: widget.thinking,
          ),
          child: SizedBox(
            width: widget.size,
            height: widget.size,
            child: Center(
              child: Icon(
                widget.listening ? Icons.stop_rounded : Icons.mic_rounded,
                color: Hud.bright,
                size: widget.size * 0.22,
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }
}

class _OrbPainter extends CustomPainter {
  _OrbPainter({required this.t, required this.listening, required this.thinking});
  final double t;
  final bool listening;
  final bool thinking;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);
    final r = size.width / 2;
    final pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi);

    // outer glow halo (stronger while listening)
    final glow = Paint()
      ..color = Hud.primary.withOpacity(listening ? 0.28 + 0.22 * pulse : 0.12)
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, 24 + 12 * pulse);
    canvas.drawCircle(c, r * 0.72, glow);

    // faint concentric rings
    for (var i = 0; i < 3; i++) {
      final rr = r * (0.55 + i * 0.16);
      canvas.drawCircle(
        c,
        rr,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1
          ..color = Hud.primary.withOpacity(0.18 - i * 0.04),
      );
    }

    // core disc
    canvas.drawCircle(
      c,
      r * 0.5,
      Paint()
        ..shader = RadialGradient(colors: [
          Hud.primary.withOpacity(listening ? 0.5 : 0.32),
          Hud.bgPanel.withOpacity(0.9),
        ]).createShader(Rect.fromCircle(center: c, radius: r * 0.5)),
    );
    // core rim
    canvas.drawCircle(
      c,
      r * 0.5,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = Hud.bright.withOpacity(listening ? 0.9 : 0.5),
    );

    // tick ring — static dashes, rotating while thinking
    const ticks = 40;
    final rot = thinking ? t * 2 * math.pi : 0.0;
    for (var i = 0; i < ticks; i++) {
      final a = rot + i / ticks * 2 * math.pi;
      final on = thinking ? ((i / ticks + t) % 1 < 0.35) : true;
      final p1 = c + Offset(math.cos(a), math.sin(a)) * (r * 0.82);
      final p2 = c + Offset(math.cos(a), math.sin(a)) * (r * 0.9);
      canvas.drawLine(
        p1,
        p2,
        Paint()
          ..strokeWidth = 2
          ..color = Hud.primary.withOpacity(on ? 0.55 : 0.12),
      );
    }
  }

  @override
  bool shouldRepaint(_OrbPainter old) =>
      old.t != t || old.listening != listening || old.thinking != thinking;
}
