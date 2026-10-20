# Flutter engine + embedding — keep so R8 doesn't strip channel/reflection entry points.
-keep class io.flutter.app.** { *; }
-keep class io.flutter.plugin.** { *; }
-keep class io.flutter.embedding.** { *; }
-keep class io.flutter.util.** { *; }
-keep class io.flutter.view.** { *; }
-keep class io.flutter.** { *; }
-dontwarn io.flutter.embedding.**

# Our native bridge (MethodChannel handlers are invoked reflectively by name).
-keep class com.ilioski.medo_phone.** { *; }

# Plugins used by MEDO (speech_to_text, flutter_tts) ship their own consumer
# rules, but keep their packages defensively — they touch platform services.
-keep class com.csdcorp.speech_to_text.** { *; }
-keep class com.tundralabs.fluttertts.** { *; }

# AndroidX FileProvider (camera capture) is referenced only from the manifest.
-keep class androidx.core.content.FileProvider { *; }

# Play Core split-install classes are referenced by Flutter's
# FlutterPlayStoreSplitApplication shim, but this app has no deferred
# components, so the library isn't present. Tell R8 that's fine.
-dontwarn com.google.android.play.core.**
-keep class io.flutter.app.FlutterPlayStoreSplitApplication { *; }
-keep class io.flutter.embedding.android.FlutterPlayStoreSplitApplication { *; }
