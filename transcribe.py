"""
GRANOLA — Transcription Google STT + timecodes par détection d'énergie.

Les trois fonctions sont exportées séparément pour que main.py
puisse les combiner librement.
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


def detect_speech_segments(audio, sr, frame_ms=20, energy_percentile=30,
                           min_duration_ms=80, min_silence_ms=100,
                           smooth_ms=60):
    """
    Détecte les segments de parole via l'enveloppe d'énergie.
    Retourne [(start_s, end_s), ...].
    """
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len

    n_frames = len(audio) // hop
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * hop: (i + 1) * hop]
        energy[i] = np.sum(frame ** 2)

    smooth_frames = max(1, int(smooth_ms / frame_ms))
    energy_smooth = uniform_filter1d(energy, size=smooth_frames)

    threshold = np.percentile(energy_smooth, energy_percentile)
    threshold = max(threshold, np.median(energy_smooth) * 0.5)

    is_speech = energy_smooth > threshold

    segments = []
    in_seg = False
    start = 0

    for i in range(len(is_speech)):
        if is_speech[i] and not in_seg:
            start = i
            in_seg = True
        elif not is_speech[i] and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(is_speech)))

    min_frames = max(1, int(min_duration_ms / frame_ms))
    segments = [(s, e) for s, e in segments if (e - s) >= min_frames]

    min_sil = max(1, int(min_silence_ms / frame_ms))
    merged = []
    for s, e in segments:
        if merged and (s - merged[-1][1]) < min_sil:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s * hop / sr, e * hop / sr) for s, e in merged]


def match_words_to_segments(words, segments):
    """
    Associe chaque mot (dans l'ordre) à un segment temporel.
    """
    if not words:
        return []
    if not segments:
        return [{"word": w, "start": 0.0, "end": 0.0} for w in words]

    result = []
    n_words = len(words)
    n_segs = len(segments)

    if n_words <= n_segs:
        for i, word in enumerate(words):
            s, e = segments[i]
            result.append({"word": word, "start": s, "end": e})
    else:
        words_per_seg = n_words / n_segs
        idx = 0
        for seg_i, (s, e) in enumerate(segments):
            next_idx = min(int(round((seg_i + 1) * words_per_seg)), n_words)
            seg_words = words[idx:next_idx]
            if not seg_words:
                continue
            dt = (e - s) / len(seg_words)
            for j, w in enumerate(seg_words):
                result.append({"word": w, "start": s + j * dt,
                               "end": s + (j + 1) * dt})
            idx = next_idx

        for w in words[idx:]:
            result.append({"word": w, "start": segments[-1][0],
                           "end": segments[-1][1]})

    return result