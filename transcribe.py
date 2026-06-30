"""Transcription avec l'API Google Speech Recognition (gratuite, sans clé)."""

import numpy as np
import tempfile
import soundfile as sf
import speech_recognition as sr_lib


def transcribe_audio(audio, sr=16000, language="fr-FR", model_size=None):
    """
    Transcrit un numpy array en texte via Google Web Speech API.

    Paramètres
    ----------
    audio      : numpy array (float)
    sr         : sample rate
    language   : code langue Google ("fr-FR", "en-US", ...)
    model_size : ignoré, gardé pour compatibilité avec main.py
    """
    # Normaliser
    audio_f32 = audio.astype(np.float32)
    peak = np.max(np.abs(audio_f32))
    if peak > 0:
        audio_f32 = audio_f32 / peak

    # Écrire un wav temporaire
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_f32, sr)

    recognizer = sr_lib.Recognizer()

    with sr_lib.AudioFile(tmp_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language=language)
    except sr_lib.UnknownValueError:
        text = ""
    except sr_lib.RequestError as e:
        print(f"    ✗ Erreur API Google : {e}")
        text = ""

    # Nettoyage
    import os
    os.unlink(tmp_path)

    return text.strip()