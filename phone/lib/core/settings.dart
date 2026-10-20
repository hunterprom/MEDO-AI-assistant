/// App settings: the cloud-chat endpoint + key, and the default weather city.
///
/// The key is stored on-device in shared_preferences and never leaves the
/// phone except in the Authorization header of the chat request you opted into.
library;

import 'package:shared_preferences/shared_preferences.dart';

class AppSettings {
  AppSettings._(this._p);
  final SharedPreferences _p;

  static AppSettings? _i;
  static AppSettings get I => _i!;
  static Future<AppSettings> init() async =>
      _i ??= AppSettings._(await SharedPreferences.getInstance());

  /// What handles open-ended questions that no on-device skill matched (and
  /// when not linked to the desktop MEDO):
  ///   'none'  — stay fully on-device; reply "add a model"
  ///   'cloud' — call the selected AI model's API and answer inline in the chat
  String get chatBackend => _p.getString('chat_backend') ?? 'cloud';
  set chatBackend(String v) => _p.setString('chat_backend', v);

  // --- AI model providers --------------------------------------------------
  // Each is reached through the same OpenAI-compatible /chat/completions call,
  // so a provider is just an endpoint + default model. The answer is returned
  // to MEDO and spoken/shown in the chat (not opened in a separate app — an
  // installed app can't hand its reply back to another app).
  static const providers = <String, ({String label, String base, String model})>{
    'groq': (
      label: 'Groq',
      base: 'https://api.groq.com/openai/v1',
      model: 'llama-3.3-70b-versatile'
    ),
    'openai': (
      label: 'ChatGPT',
      base: 'https://api.openai.com/v1',
      model: 'gpt-4o-mini'
    ),
    'anthropic': (
      label: 'Claude',
      base: 'https://api.anthropic.com/v1',
      model: 'claude-3-5-haiku-latest'
    ),
    'deepseek': (
      label: 'DeepSeek',
      base: 'https://api.deepseek.com/v1',
      model: 'deepseek-chat'
    ),
    'gemini': (
      label: 'Gemini',
      base: 'https://generativelanguage.googleapis.com/v1beta/openai',
      model: 'gemini-1.5-flash'
    ),
    'custom': (label: 'Custom', base: '', model: ''),
  };

  /// Which provider handles cloud chat.
  String get chatProvider => _p.getString('chat_provider') ?? 'groq';
  set chatProvider(String v) => _p.setString('chat_provider', v);

  // A default Groq key can be baked into the build with
  // `--dart-define=GROQ_KEY=…`. It is NOT stored in source (so it never lands in
  // git); only the compiled APK carries it. A key typed in Settings overrides it.
  static const _groqDefaultKey = String.fromEnvironment('GROQ_KEY');

  // Keys and model overrides are stored PER provider, so switching between
  // ChatGPT/Claude/… doesn't wipe the others' credentials.
  String providerKey(String id) {
    final stored = _p.getString('key_$id') ?? '';
    if (stored.isNotEmpty) return stored;
    if (id == 'groq') return _groqDefaultKey; // build-time fallback
    return '';
  }

  void setProviderKey(String id, String v) => _p.setString('key_$id', v.trim());

  String providerModel(String id) {
    final override = _p.getString('model_$id') ?? '';
    if (override.isNotEmpty) return override;
    return providers[id]?.model ?? '';
  }

  void setProviderModel(String id, String v) => _p.setString('model_$id', v.trim());

  // Custom provider's editable endpoint.
  String get customBase => _p.getString('custom_base') ?? '';
  set customBase(String v) => _p.setString('custom_base', v.trim());

  /// The endpoint/model/key actually used for the current selection.
  String get effectiveBaseUrl =>
      chatProvider == 'custom' ? customBase : (providers[chatProvider]?.base ?? '');
  String get effectiveModel => providerModel(chatProvider);
  String get effectiveKey => providerKey(chatProvider);

  bool get chatEnabled => effectiveKey.isNotEmpty && effectiveBaseUrl.isNotEmpty;

  // Vision-capable model per provider, used when the assistant looks through the
  // camera. Empty => that provider can't see images over its API. (Custom is
  // assumed to point at a vision model — we use whatever model is set.)
  static const _visionModels = <String, String>{
    'groq': 'meta-llama/llama-4-scout-17b-16e-instruct',
    'openai': 'gpt-4o-mini',
    'anthropic': 'claude-3-5-sonnet-latest',
    'deepseek': '', // no vision model on the OpenAI-compatible API
    'gemini': 'gemini-1.5-flash',
  };

  /// The model to use for camera/vision, for the current provider.
  String get visionModel =>
      chatProvider == 'custom' ? effectiveModel : (_visionModels[chatProvider] ?? '');

  bool get canSee => chatEnabled && visionModel.isNotEmpty;

  // Weather default location.
  String get city => _p.getString('city') ?? 'Skopje';
  set city(String v) => _p.setString('city', v.trim());

  // Personality name shown in the UI.
  String get assistantName => _p.getString('name') ?? 'MEDO';

  // --- connect to MEDO (desktop) — same system as the watch app ------------
  // When linked, utterances are routed to the desktop MEDO's companion API
  // instead of the on-device skills. `medoAddress` empty => not linked.
  String get medoAddress => _p.getString('medo_addr') ?? '';
  set medoAddress(String v) => _p.setString('medo_addr', v.trim());

  String get medoToken => _p.getString('medo_token') ?? '';
  set medoToken(String v) => _p.setString('medo_token', v.trim());

  /// Route to the desktop MEDO (true) or run on-device (false). Only meaningful
  /// when [medoAddress] is set; the UI toggle writes this.
  bool get useMedo => (_p.getBool('use_medo') ?? false) && medoAddress.isNotEmpty;
  set useMedo(bool v) => _p.setBool('use_medo', v);

  bool get linked => medoAddress.isNotEmpty;
}
