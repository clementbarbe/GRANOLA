"""Débruitage audio IRM par filtre de Wiener avec lissage."""

import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d, median_filter

# Paramètres STFT partagés
NPERSEG = 1024
NOVERLAP = 768


def estimate_noise_profile(noise_audio, sr, nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Estime le profil spectral du bruit (magnitude² moyenne par bin).
    """
    _, _, Z = stft(noise_audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    noise_power = np.mean(np.abs(Z) ** 2, axis=1)
    return noise_power


def spectral_subtraction(audio, sr, noise_profile, alpha=2.0, beta=0.02,
                         time_smooth=5, freq_smooth=3,
                         nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Filtre de Wiener avec lissage temporel et spectral.

    Au lieu de soustraire directement le spectre de bruit, on calcule
    un masque de gain G(f,t) = max(1 - α·Pn/|X|², β) puis on lisse
    ce masque pour éviter le bruit musical.

    Paramètres
    ----------
    audio          : signal à débruiter
    noise_profile  : puissance moyenne du bruit par bin (de estimate_noise_profile)
    alpha          : facteur de sursoustraction (>1 = plus agressif)
    beta           : plancher du gain (0 < beta < 1)
    time_smooth    : largeur du lissage temporel du masque (en trames)
    freq_smooth    : largeur du lissage fréquentiel du masque (en bins)
    """
    _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Z) ** 2
    phase = np.angle(Z)

    # ── Masque de Wiener ──
    noise_2d = noise_profile[:, np.newaxis]
    snr = power / (noise_2d + 1e-10)
    gain = np.maximum(1.0 - alpha / (snr + 1e-10), beta)

    # ── Lissage du masque pour supprimer le bruit musical ──
    # Lissage temporel (axe 1 = trames)
    if time_smooth > 1:
        gain = uniform_filter1d(gain, size=time_smooth, axis=1)
    # Lissage fréquentiel (axe 0 = bins)
    if freq_smooth > 1:
        gain = uniform_filter1d(gain, size=freq_smooth, axis=0)
    # Filtre médian pour enlever les pics isolés
    gain = median_filter(gain, size=(3, 3))

    # Clamp final
    gain = np.clip(gain, beta, 1.0)

    # ── Reconstruction ──
    mag_clean = np.sqrt(power) * gain
    Z_clean = mag_clean * np.exp(1j * phase)
    _, audio_clean = istft(Z_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # Ajuster la longueur
    if len(audio_clean) >= len(audio):
        audio_clean = audio_clean[:len(audio)]
    else:
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))

    return audio_clean