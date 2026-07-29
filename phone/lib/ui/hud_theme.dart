import 'package:flutter/material.dart';

/// The MEDO HUD palette + type, lifted from the desktop arc-reactor HUD so the
/// phone reads as the same product: deep cosmic background, cyan glow, and a
/// mono, letter-spaced label style.
class Hud {
  static const bgDeep = Color(0xFF04070D);
  static const bgPanel = Color(0xFF071224);
  static const panelBorder = Color(0x3338A8FF); // cyan @ 20%
  static const primary = Color(0xFF38A8FF);
  static const bright = Color(0xFF7FD0FF);
  static const text = Color(0xFFC9E6FF);
  static const muted = Color(0xFF4A6D94);
  static const danger = Color(0xFFFF7A6B);
  static const ok = Color(0xFF6EE7A8);

  /// Uppercase, spaced, mono — the HUD's chrome label look.
  static TextStyle label({double size = 11, Color color = muted, double spacing = 2}) =>
      TextStyle(
        fontFamily: 'monospace',
        fontSize: size,
        letterSpacing: spacing,
        color: color,
        fontWeight: FontWeight.w500,
      );

  static const cosmic = BoxDecoration(
    gradient: RadialGradient(
      center: Alignment(0, -0.4),
      radius: 1.3,
      colors: [Color(0xFF0A1526), Color(0xFF04070D)],
    ),
  );
}
