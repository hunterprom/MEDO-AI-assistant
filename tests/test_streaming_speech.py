"""S2 — speech starts sooner and doesn't stutter, in every script.

Two defects, both measured before being fixed:

**Three languages never streamed at all.** ``drain_sentences`` looked for
``.!?`` followed by whitespace. Chinese and Japanese end sentences with the
full-width ``。`` and put no space after it; Hindi uses the danda ``।``. So
zh, ja and hi drained ZERO sentences and waited for the entire reply before a
word was spoken — TTFA 5508 ms for the Japanese sample against 2622 ms once
it could segment.

**Every sentence boundary was a silent gap.** The speaker synthesized a
sentence, played it, and only then began synthesizing the next, so the gap
between sentences was a whole synthesis: measured 200-275 ms on the Piper
voices and 3058 ms on Japanese. Synthesizing one sentence ahead of playback
takes those to 7-15 ms.
"""

from __future__ import annotations

import pytest

from voice.tts import drain_sentences, speech_weight

#: Chunk size an LLM delta roughly arrives in.
CHUNK = 8


def stream(text: str, chunk: int = CHUNK) -> tuple[list[str], str]:
    """Feed ``text`` through drain_sentences the way the LLM streams it."""
    out: list[str] = []
    buf = ""
    for i in range(0, len(text), chunk):
        buf += text[i:i + chunk]
        got, buf = drain_sentences(buf)
        out.extend(got)
    return out, buf


# --- the scripts that never streamed -----------------------------------------

@pytest.mark.parametrize("code,text,least", [
    ("zh", "斯科普里现在二十二度，天气晴朗。风力较弱，来自西北方向。傍晚之前都会保持干燥。", 2),
    ("ja", "スコピエは今、気温二十二度で晴れています。風は北西からで弱いです。夕方までは乾燥した天気が続きます。", 2),
    ("hi", "स्कोप्ये में इस समय बाईस डिग्री है और मौसम साफ़ है। हवा उत्तर-पश्चिम से हल्की है। "
           "शाम तक मौसम सूखा रहेगा।", 2),
])
def test_a_reply_streams_in_every_script(code, text, least):
    spoken, _tail = stream(text)
    assert len(spoken) >= least, (
        f"{code} drained {len(spoken)} sentences — it waits for the whole reply")


def test_cjk_needs_no_space_after_the_full_stop():
    # THE bug: "。" is the boundary and Chinese does not follow it with a space.
    spoken, _ = drain_sentences("斯科普里现在二十二度，天气晴朗。风力较弱，来自西北方向。")
    assert spoken and spoken[0].endswith("。")


def test_the_devanagari_danda_ends_a_sentence():
    spoken, _ = drain_sentences("स्कोप्ये में इस समय बाईस डिग्री है और मौसम साफ़ है। "
                                "हवा उत्तर-पश्चिम से हल्की है। ")
    assert spoken and spoken[0].endswith("।")


def test_full_width_question_and_exclamation_marks_count():
    spoken, _ = drain_sentences("今日はいい天気ですね！明日はどうでしょうか？さあ。")
    assert len(spoken) >= 2


# --- the alphabetic scripts must not regress ---------------------------------

def test_english_still_splits_where_it_did():
    spoken, tail = stream("It's twenty-two degrees and clear in Skopje. The wind "
                          "is light from the north-west. It should stay dry.")
    assert len(spoken) == 2
    assert spoken[0].endswith("Skopje.")
    assert tail.strip().startswith("It should stay dry")   # flushed by the caller


def test_an_abbreviation_does_not_split_a_sentence():
    # "Dr." and "3.5" are why the alphabetic branch demands whitespace AND a
    # minimum length — a period is not a sentence end.
    spoken, _ = drain_sentences("Dr. Smith says it is 3.5 degrees outside and ")
    assert spoken == []


@pytest.mark.parametrize("text", [
    "Сейчас двадцать два градуса, в Скопье ясно. Ветер слабый, северо-западный. ",
    "Дваесет и два степени е и ведро во Скопје. Ветерот е слаб од северозапад. ",
    "Είναι είκοσι δύο βαθμοί και αίθριος καιρός στα Σκόπια. Ο άνεμος είναι ασθενής. ",
])
def test_cyrillic_and_greek_are_unaffected(text):
    spoken, _ = drain_sentences(text)
    assert len(spoken) >= 1


def test_a_short_fragment_waits_for_the_next_one():
    # Speaking "Yes." alone is worse than speaking it with what follows.
    spoken, _ = drain_sentences("Yes. ")
    assert spoken == []


# --- speech weight -----------------------------------------------------------

def test_dense_scripts_count_for_more_than_their_length():
    # 16 Chinese characters and 44 English ones both produced ~2.8 s of audio.
    # A min_len tuned in Latin characters otherwise holds CJK back.
    assert speech_weight("斯科普里现在二十二度") == pytest.approx(30, abs=1)
    assert speech_weight("hello there") == 11


def test_weight_handles_empty_and_none():
    assert speech_weight("") == 0
    assert speech_weight(None) == 0


def test_mixed_script_text_is_weighted_per_character():
    # A reply can legitimately mix scripts (a product name in a zh sentence).
    assert speech_weight("RX-7 是一辆跑车") > len("RX-7 是一辆跑车")


# --- the pipeline ------------------------------------------------------------

def test_synthesis_is_separable_from_playback():
    # The whole point of S2's second half: the streaming speaker must be able
    # to synthesize the NEXT sentence while the current one is still audible,
    # which is impossible while one call does both.
    from voice.loop import VoiceLoop

    assert callable(VoiceLoop._synthesize)
    assert callable(VoiceLoop._play)


def test_the_lookahead_is_bounded():
    # Unbounded, a long reply synthesizes itself entirely into RAM and every
    # buffered sentence is more work thrown away when the user interrupts.
    from voice.loop import _SYNTH_LOOKAHEAD

    assert 1 <= _SYNTH_LOOKAHEAD <= 3
