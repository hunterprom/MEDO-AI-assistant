/// Cloud chat fallback — an OpenAI-compatible client (Groq by default).
///
/// Only used when no on-device skill matched AND the user configured a key.
/// One short system prompt keeps replies speakable (plain text, no markdown).
library;

import 'dart:convert';

import 'package:http/http.dart' as http;

import 'settings.dart';

class LlmException implements Exception {
  const LlmException(this.message);
  final String message;
  @override
  String toString() => message;
}

class LlmClient {
  static const _timeout = Duration(seconds: 30);

  /// Ask the cloud model one question with a little rolling context.
  /// [history] is prior (userText, replyText) turns, oldest first.
  Future<String> chat(String text, List<(String, String)> history) async {
    final s = AppSettings.I;
    if (!s.chatEnabled) {
      throw const LlmException(
          'No AI model is set up — pick one and add its key in settings.');
    }
    final messages = <Map<String, String>>[
      {
        'role': 'system',
        'content':
            'You are ${s.assistantName}, a concise voice assistant on a phone. '
                'Answer in one to three spoken sentences, plain text, no '
                'markdown or lists. If you are unsure, say so briefly.'
      },
      for (final (u, a) in history) ...[
        {'role': 'user', 'content': u},
        {'role': 'assistant', 'content': a},
      ],
      {'role': 'user', 'content': text},
    ];

    final uri = Uri.parse('${s.effectiveBaseUrl}/chat/completions');
    late http.Response resp;
    try {
      resp = await http
          .post(
            uri,
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer ${s.effectiveKey}',
            },
            body: jsonEncode({
              'model': s.effectiveModel,
              'messages': messages,
              'temperature': 0.5,
              'max_tokens': 300,
            }),
          )
          .timeout(_timeout);
    } catch (_) {
      throw const LlmException(
          "I couldn't reach the cloud model — check your connection.");
    }
    if (resp.statusCode == 401) {
      throw const LlmException('The cloud API rejected the key — check it in settings.');
    }
    if (resp.statusCode >= 400) {
      throw LlmException('The cloud model returned an error (${resp.statusCode}).');
    }
    return _extractContent(resp.body);
  }

  /// Describe a captured photo. [base64Jpeg] is the raw base64 (no data prefix).
  /// Uses the provider's vision model; throws if the provider can't see.
  Future<String> see(String prompt, String base64Jpeg) async {
    final s = AppSettings.I;
    if (!s.chatEnabled) {
      throw const LlmException(
          'To see, pick an AI model with vision (ChatGPT, Claude, or Gemini) in settings.');
    }
    if (s.visionModel.isEmpty) {
      final label = AppSettings.providers[s.chatProvider]?.label ?? 'This model';
      throw LlmException(
          "$label can't see images. Switch to ChatGPT, Claude, or Gemini in settings.");
    }
    final messages = <Map<String, Object?>>[
      {
        'role': 'system',
        'content':
            'You are ${s.assistantName}, describing what the phone camera sees. '
                'Answer in one to three spoken sentences, plain text, no markdown. '
                'Be direct about the main things in view.'
      },
      {
        'role': 'user',
        'content': [
          {
            'type': 'text',
            'text': prompt.trim().isEmpty ? 'What do you see?' : prompt.trim()
          },
          {
            'type': 'image_url',
            'image_url': {'url': 'data:image/jpeg;base64,$base64Jpeg'}
          },
        ],
      },
    ];

    final uri = Uri.parse('${s.effectiveBaseUrl}/chat/completions');
    late http.Response resp;
    try {
      resp = await http
          .post(
            uri,
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer ${s.effectiveKey}',
            },
            body: jsonEncode({
              'model': s.visionModel,
              'messages': messages,
              'temperature': 0.4,
              'max_tokens': 300,
            }),
          )
          .timeout(const Duration(seconds: 45));
    } catch (_) {
      throw const LlmException(
          "I couldn't reach the model to describe that — check your connection.");
    }
    if (resp.statusCode == 401) {
      throw const LlmException('The API rejected the key — check it in settings.');
    }
    if (resp.statusCode >= 400) {
      throw LlmException(
          'The vision model returned an error (${resp.statusCode}). Its name may be wrong for this provider.');
    }
    return _extractContent(resp.body, empty: "I couldn't make out what's in view.");
  }

  String _extractContent(String responseBody,
      {String empty = "I didn't get an answer to that."}) {
    try {
      final body = jsonDecode(responseBody) as Map<String, Object?>;
      final choices = body['choices'] as List<Object?>?;
      final msg = (choices?.first as Map<String, Object?>?)?['message']
          as Map<String, Object?>?;
      final content = (msg?['content'] as String?)?.trim() ?? '';
      return content.isEmpty ? empty : content;
    } catch (_) {
      throw const LlmException('The model sent an unexpected reply.');
    }
  }
}
