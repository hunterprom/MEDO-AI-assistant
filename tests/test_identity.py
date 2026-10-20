"""Local speaker verification (S2): enrol a voiceprint, verify a sample.

A fake embedder (identity on vectors) drives the logic without any ML model, so
these pin the CONTRACT: only the embedding is stored (never audio), verification
is cosine-vs-threshold, and every degenerate path (no embedder, not enrolled,
embed error, low similarity) fails CLOSED to False.
"""

from __future__ import annotations

import json

import pytest

from security.identity import SpeakerVerifier, VoiceprintStore, cosine


def _store(tmp_path):
    return VoiceprintStore(tmp_path / "voiceprint.json")


def _identity_embedder(sample):
    # The "audio" IS the embedding vector — lets tests control similarity exactly.
    return list(sample)


# -- cosine -------------------------------------------------------------------

def test_cosine_edges():
    assert cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert cosine([], [1]) == -1.0                 # empty
    assert cosine([1, 2], [1, 2, 3]) == -1.0       # ragged
    assert cosine([0, 0], [1, 1]) == -1.0          # zero vector never clears


# -- store: embedding only, owner-only, round-trips ---------------------------

def test_store_persists_only_the_embedding(tmp_path):
    st = _store(tmp_path)
    assert st.load() is None and not st.exists()
    st.save([0.1, 0.2, 0.3])
    assert st.exists() and st.load() == [0.1, 0.2, 0.3]
    raw = json.loads((tmp_path / "voiceprint.json").read_text("utf-8"))
    assert set(raw) == {"voiceprint"}             # no audio, nothing else
    st.clear()
    assert st.load() is None


def test_store_survives_corrupt_file(tmp_path):
    p = tmp_path / "voiceprint.json"
    p.write_text("not json {", encoding="utf-8")
    assert VoiceprintStore(p).load() is None       # never raises


# -- verifier: availability + enrolment ---------------------------------------

def test_unavailable_without_an_embedder(tmp_path):
    v = SpeakerVerifier(_store(tmp_path), embedder=None)
    assert v.is_available() is False
    assert v.enroll([[1, 0, 0]]) is False          # can't enrol without a model
    assert v.verify([1, 0, 0]) is False            # fail closed


def test_enroll_then_verify_accepts_owner_rejects_other(tmp_path):
    v = SpeakerVerifier(_store(tmp_path), embedder=_identity_embedder,
                        threshold=0.75)
    assert v.enroll([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]) is True
    assert v.is_enrolled()
    assert v.verify([1.0, 0.05, 0.0]) is True       # close -> owner
    assert v.verify([0.0, 1.0, 0.0]) is False       # orthogonal -> not owner


def test_verify_fails_closed_when_not_enrolled(tmp_path):
    v = SpeakerVerifier(_store(tmp_path), embedder=_identity_embedder)
    assert v.is_enrolled() is False
    assert v.verify([1.0, 0.0, 0.0]) is False


def test_verify_fails_closed_when_embed_raises(tmp_path):
    def boom(_sample):
        raise RuntimeError("model exploded")

    st = _store(tmp_path)
    st.save([1.0, 0.0, 0.0])                        # enrolled...
    v = SpeakerVerifier(st, embedder=boom)
    assert v.verify([1.0, 0.0, 0.0]) is False        # ...but a crash denies


def test_threshold_is_enforced(tmp_path):
    st = _store(tmp_path)
    st.save([1.0, 0.0])
    strict = SpeakerVerifier(st, embedder=_identity_embedder, threshold=0.99)
    lax = SpeakerVerifier(st, embedder=_identity_embedder, threshold=0.5)
    sample = [0.8, 0.2]                              # cosine ~0.97
    assert strict.verify(sample) is False
    assert lax.verify(sample) is True


def test_enroll_refuses_ragged_embeddings(tmp_path):
    v = SpeakerVerifier(_store(tmp_path), embedder=_identity_embedder)
    assert v.enroll([[1, 0, 0], [1, 0]]) is False    # inconsistent -> nothing stored
    assert v.is_enrolled() is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
