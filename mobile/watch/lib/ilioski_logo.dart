/// The ILIOSKI monogram — drawn, not shipped as an asset, so it stays crisp
/// at every watch density and can glow with the app's accent color.
///
/// Design: an arc-reactor style badge (a nod to Medo) — two cyan rings with
/// a gap at the top, holding a geometric letter "I" built from three bars.
library;

import 'dart:math' as math;

import 'package:flutter/material.dart';

class IlioskiLogo extends StatelessWidget {
  const IlioskiLogo({super.key, this.size = 64, this.glow = false});

  final double size;

  /// Adds a soft outer glow (used while Medo is speaking).
  final bool glow;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size.square(size),
      painter: _LogoPainter(glow: glow),
    );
  }
}

class _LogoPainter extends CustomPainter {
  const _LogoPainter({required this.glow});

  final bool glow;

  static const _cyan = Color(0xFF18FFFF); // cyanAccent
  static const _teal = Color(0xFF00BFA5); // tealAccent-ish

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final s = size.shortestSide;

    final gradient = const SweepGradient(
      colors: [_teal, _cyan, _teal],
      stops: [0.0, 0.5, 1.0],
      transform: GradientRotation(-math.pi / 2),
    ).createShader(Rect.fromCircle(center: center, radius: s / 2));

    if (glow) {
      final glowPaint = Paint()
        ..color = _cyan.withOpacity(0.35)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
      canvas.drawCircle(center, s * 0.46, glowPaint);
    }

    // Outer ring, broken at the top like a reactor housing.
    final ring = Paint()
      ..shader = gradient
      ..style = PaintingStyle.stroke
      ..strokeWidth = s * 0.055
      ..strokeCap = StrokeCap.round;
    const gap = math.pi / 5;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: s * 0.44),
      -math.pi / 2 + gap / 2,
      2 * math.pi - gap,
      false,
      ring,
    );

    // Inner ring, fainter.
    final innerRing = Paint()
      ..color = _cyan.withOpacity(0.30)
      ..style = PaintingStyle.stroke
      ..strokeWidth = s * 0.028;
    canvas.drawCircle(center, s * 0.33, innerRing);

    // The "I": top bar, stem, bottom bar.
    final bar = Paint()..shader = gradient;
    RRect rr(double cx, double cy, double w, double h) =>
        RRect.fromRectAndRadius(
          Rect.fromCenter(
            center: Offset(cx, cy),
            width: w,
            height: h,
          ),
          Radius.circular(s * 0.02),
        );
    canvas.drawRRect(rr(center.dx, center.dy - s * 0.155, s * 0.26, s * 0.075), bar);
    canvas.drawRRect(rr(center.dx, center.dy, s * 0.085, s * 0.24), bar);
    canvas.drawRRect(rr(center.dx, center.dy + s * 0.155, s * 0.26, s * 0.075), bar);
  }

  @override
  bool shouldRepaint(_LogoPainter oldDelegate) => oldDelegate.glow != glow;
}
