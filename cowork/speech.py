"""Everything Cowork says, in both languages, in one place.

MEDO answers in the language it was asked in. A plan is a long conversation —
summary, approval, per-step narration, a failure and its options, a final
report — and every one of those lines is a chance to drop back into English
halfway through a Macedonian task. Keeping them in one table is what makes that
checkable: the test asserts every key has both languages, so a new line cannot
ship English-only.
"""

from __future__ import annotations

#: key -> {lang: template}. ``{}``-formatted with keyword arguments.
PHRASES: dict[str, dict[str, str]] = {
    # -- approval ------------------------------------------------------------
    "planning": {
        # Measured on this machine: qwen3:30b takes 60-120 s to plan against a
        # 69-capability catalogue. Silence that long reads as MEDO having
        # ignored the request, and the user says it again.
        "en": "Let me work out a plan.",
        "mk": "Само да составам план.",
    },
    "approved": {
        "en": "Right — starting.",
        "mk": "Добро — почнувам.",
    },
    "cancelled": {
        "en": "Okay, dropped it.",
        "mk": "Добро, откажано.",
    },
    "not_understood": {
        "en": "Say yes to run it, no to drop it, or tell me which step to skip.",
        "mk": "Кажи да за да го извршам, не за да го откажам, или кажи кој "
              "чекор да го прескокнам.",
    },
    "step_removed": {
        "en": "Dropped step {number}. {summary}",
        "mk": "Го тргнав чекорот {number}. {summary}",
    },
    "step_changed": {
        "en": "Changed step {number}. {summary}",
        "mk": "Го променив чекорот {number}. {summary}",
    },
    "no_such_step": {
        "en": "There's no step {number}.",
        "mk": "Нема чекор {number}.",
    },
    # -- execution -----------------------------------------------------------
    "step_done": {
        "en": "Step {number} done: {speech}",
        "mk": "Чекор {number} готов: {speech}",
    },
    "step_done_quiet": {
        "en": "Step {number} done.",
        "mk": "Чекор {number} готов.",
    },
    "step_skipped": {
        "en": "Skipping step {number} — {reason}.",
        "mk": "Го прескокнувам чекорот {number} — {reason}.",
    },
    "step_failed": {
        "en": "Step {number} failed: {reason}. Retry, skip it, or stop?",
        "mk": "Чекорот {number} не успеа: {reason}. Да пробам пак, да "
              "прескокнам, или да запрам?",
    },
    "step_confirm": {
        "en": "Step {number}: {prompt}",
        "mk": "Чекор {number}: {prompt}",
    },
    "step_not_understood": {
        "en": "Yes or no — shall I do step {number}?",
        "mk": "Да или не — да го направам чекорот {number}?",
    },
    "generic_confirm": {
        "en": "Shall I {description}?",
        "mk": "Да го направам ова: {description}?",
    },
    "finished": {
        "en": "Task finished — {progress}.",
        "mk": "Задачата е завршена — {progress}.",
    },
    "finished_with_gaps": {
        "en": "Task finished, but not all of it — {progress}. {gaps}",
        "mk": "Задачата заврши, но не цела — {progress}. {gaps}",
    },
    "stopped": {
        "en": "Stopped at step {number}. Nothing after it ran.",
        "mk": "Застанав на чекорот {number}. Ништо после него не се изврши.",
    },
    "aborted": {
        "en": "Stopped. {progress}.",
        "mk": "Запрено. {progress}.",
    },
    "options": {
        "en": "Say retry, skip, or stop.",
        "mk": "Кажи повтори, прескокни, или запри.",
    },
    "nothing_running": {
        "en": "There's no task running.",
        "mk": "Нема задача во тек.",
    },
    # -- undo ----------------------------------------------------------------
    "undo_offer": {
        "en": "I can undo step {number} — want me to?",
        "mk": "Можам да го вратам чекорот {number} — да го сторнирам?",
    },
    "undo_done": {
        "en": "Undid step {number}.",
        "mk": "Го вратив чекорот {number}.",
    },
    "undo_impossible": {
        "en": "I can't undo what already ran — {what}.",
        "mk": "Не можам да го вратам тоа што веќе се изврши — {what}.",
    },
    # -- refusals ------------------------------------------------------------
    "unattended": {
        "en": "I won't run a task like this on a schedule — it needs someone "
              "to approve the plan.",
        "mk": "Нема да извршам ваква задача по распоред — треба некој да го "
              "одобри планот.",
    },
    "disabled": {
        "en": "Cowork is switched off.",
        "mk": "Cowork е исклучен.",
    },
    "dry_run": {
        "en": "Dry run — I won't actually do any of it. {summary}",
        "mk": "Проба — нема ништо навистина да направам. {summary}",
    },
    "denied": {
        "en": "Step {number} isn't allowed: {reason}.",
        "mk": "Чекорот {number} не е дозволен: {reason}.",
    },
    # -- saving a plan as a routine (S4) -------------------------------------
    "saved": {
        "en": "Saved as the {name} routine — say it any time.",
        "mk": "Зачувано како рутина {name} — кажи го тоа кога сакаш.",
    },
    "save_refused": {
        "en": "I can't save that as a routine: {reason}.",
        "mk": "Не можам да го зачувам тоа како рутина: {reason}.",
    },
    "nothing_to_save": {
        "en": "There's no plan to save.",
        "mk": "Нема план за зачувување.",
    },
}


def say(key: str, language: str = "en", **kwargs: object) -> str:
    """One phrase, in ``language``, formatted.

    Falls back to English for an unknown language rather than to nothing: a
    missing translation should sound wrong, not be silent.
    """
    bank = PHRASES.get(key)
    if not bank:
        return ""
    template = bank.get(language) or bank.get("en") or ""
    try:
        return template.format(**kwargs).strip()
    except (KeyError, IndexError):
        return template.strip()
