"""
GRANOLA — Transcription Google STT + timecodes par détection d'énergie.

Contexte : mots isolés (~800 ms chacun) dans des runs de ~15s.
"""

import numpy as np
import tempfile
import os
import soundfile as sf
import speech_recognition as sr_lib
from scipy.ndimage import uniform_filter1d


def transcribe_google(audio, sr, language="fr-FR"):
    """Google STT → liste de mots (strings)."""
    audio_f32 = audio.astype(np.float32)
    peak = np.max(np.abs(audio_f32))
    if peak > 0:
        audio_f32 = audio_f32 / peak

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_f32, sr)
    tmp.close()

    recognizer = sr_lib.Recognizer()
    with sr_lib.AudioFile(tmp.name) as source:
        audio_data = recognizer.record(source)
    os.unlink(tmp.name)

    try:
        text = recognizer.recognize_google(audio_data, language=language)
    except sr_lib.UnknownValueError:
        return []
    except sr_lib.RequestError as e:
        print(f"    ✗ Erreur API Google : {e}")
        return []

    return text.strip().split()


def detect_speech_segments(audio, sr, n_words_hint=None):
    """
    Détecte les segments de parole par enveloppe d'énergie.

    Approche simple et robuste :
    1. Calculer l'enveloppe d'énergie lissée (fenêtres de 50 ms)
    2. Seuil = percentile adaptatif
    3. Fusionner tout ce qui est à moins de 200 ms
    4. Jeter tout ce qui fait moins de 400 ms
    5. Si on connaît le nombre de mots, ajuster le seuil
       pour obtenir le bon nombre de segments
    """
    # ── Enveloppe d'énergie ──
    frame_ms = 10
    hop = int(sr * frame_ms / 1000)
    n_frames = len(audio) // hop

    energy = np.array([
        np.sum(audio[i * hop:(i + 1) * hop] ** 2)
        for i in range(n_frames)
    ])

    # Lissage large (50 ms) pour ignorer les micro-fluctuations
    smooth = max(1, int(50 / frame_ms))
    energy = uniform_filter1d(energy, size=smooth)

    # ── Recherche du bon seuil ──
    # On balaye le percentile pour trouver le bon nombre de segments
    min_seg_ms = 400
    min_silence_ms = 200
    min_seg_frames = max(1, int(min_seg_ms / frame_ms))
    min_sil_frames = max(1, int(min_silence_ms / frame_ms))

    def segments_at_percentile(pct):
        thr = np.percentile(energy, pct)
        is_speech = energy > thr

        # Extraire segments
        segs = []
        in_seg = False
        start = 0
        for k in range(len(is_speech)):
            if is_speech[k] and not in_seg:
                start = k
                in_seg = True
            elif not is_speech[k] and in_seg:
                segs.append((start, k))
                in_seg = False
        if in_seg:
            segs.append((start, len(is_speech)))

        # Fusionner proches
        merged = []
        for s, e in segs:
            if merged and (s - merged[-1][1]) < min_sil_frames:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        # Filtrer courts
        merged = [(s, e) for s, e in merged if (e - s) >= min_seg_frames]

        return merged

    if n_words_hint and n_words_hint > 0:
        # On cherche le percentile qui donne le bon nombre de segments
        best_segs = None
        best_diff = float('inf')

        for pct in range(30, 85, 2):
            segs = segments_at_percentile(pct)
            diff = abs(len(segs) - n_words_hint)
            if diff < best_diff:
                best_diff = diff
                best_segs = segs
            if diff == 0:
                break

        segments = best_segs
    else:
        # Pas d'indice → percentile 60 (conservateur)
        segments = segments_at_percentile(60)

    # Convertir en secondes
    return [(s * hop / sr, e * hop / sr) for s, e in segments]


def match_words_to_segments(words, segments):
    """
    Associe chaque mot à un segment, dans l'ordre.

    Si même nombre → un pour un.
    Si plus de segments → on prend les N plus longs.
    Si plus de mots → on répartit dans les segments.
    """
    if not words:
        return []
    if not segments:
        return [{"word": w, "start": 0.0, "end": 0.0} for w in words]

    n_words = len(words)
    n_segs = len(segments)

    if n_words == n_segs:
        # Cas idéal
        return [{"word": w, "start": s, "end": e}
                for w, (s, e) in zip(words, segments)]

    if n_words < n_segs:
        # Plus de segments que de mots → garder les N plus longs
        durations = [(e - s, i) for i, (s, e) in enumerate(segments)]
        durations.sort(reverse=True)
        keep_idx = sorted([idx for _, idx in durations[:n_words]])
        return [{"word": w, "start": segments[idx][0], "end": segments[idx][1]}
                for w, idx in zip(words, keep_idx)]

    # Plus de mots que de segments → répartir
    result = []
    words_per_seg = n_words / n_segs
    idx = 0
    for seg_i, (s, e) in enumerate(segments):
        next_idx = min(int(round((seg_i + 1) * words_per_seg)), n_words)
        seg_words = words[idx:next_idx]
        if not seg_words:
            continue
        dt = (e - s) / len(seg_words)
        for j, w in enumerate(seg_words):
            result.append({"word": w,
                           "start": s + j * dt,
                           "end": s + (j + 1) * dt})
        idx = next_idx

    for w in words[idx:]:
        result.append({"word": w,
                       "start": segments[-1][0],
                       "end": segments[-1][1]})

    return result