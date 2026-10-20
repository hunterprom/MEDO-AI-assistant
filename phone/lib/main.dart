import 'package:flutter/material.dart';

import 'core/settings.dart';
import 'core/store.dart';
import 'ui/home_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Store.init();
  await AppSettings.init();
  runApp(const MedoApp());
}

class MedoApp extends StatelessWidget {
  const MedoApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF38A8FF);
    return MaterialApp(
      title: 'MEDO',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: seed,
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF04070D),
        snackBarTheme: const SnackBarThemeData(behavior: SnackBarBehavior.floating),
      ),
      home: const HomeScreen(),
    );
  }
}
