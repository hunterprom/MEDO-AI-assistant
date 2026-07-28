/// Medo v2 watch companion — entry point.
///
/// Speak to the watch; the transcript is routed to the Medo v2 companion
/// API on your computer (see `remote/server.py` in the repo root), and the
/// reply comes back on screen and through the speaker.
library;

import 'package:flutter/material.dart';

import 'home_screen.dart';

void main() => runApp(const MedoWatchApp());

class MedoWatchApp extends StatelessWidget {
  const MedoWatchApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MEDO',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        useMaterial3: true,
        scaffoldBackgroundColor: Colors.black,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.cyan,
          brightness: Brightness.dark,
        ),
        // Watch screens are tiny — keep tap targets compact everywhere.
        visualDensity: VisualDensity.compact,
      ),
      home: const HomeScreen(),
    );
  }
}
