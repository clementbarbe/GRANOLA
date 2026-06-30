"""
GRANOLA — GRAdient NOise Less Audio
Débruitage audio IRM par filtre de Wiener avec lissage.
"""

import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d, median_filter

NPERSEG = 1024
NOVERLAP = 768

# Paramètres optimaux trouvés par grille de recherche
DEFAULT_ALPHA = 0.8
DEFAULT_BETA = 0.0002
DEFAULT_TIME_SMOOTH = 3
DEFAULT_FREQ_SMOOTH = 3


def estimate_noise_profile(noise_audio, sr, nperseg=NPERSEG, noverlap=NOVERLAP):
    """Estime le profil spectral du bruit (magnitude² moyenne par bin)."""
    _, _, Z = stft(noise_audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return np.mean(np.abs(Z) ** 2, axis=1)


def spectral_subtraction(audio, sr, noise_profile,
                         alpha=DEFAULT_ALPHA,
                         beta=DEFAULT_BETA,
                         time_smooth=DEFAULT_TIME_SMOOTH,
                         freq_smooth=DEFAULT_FREQ_SMOOTH,
                         nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Filtre de Wiener avec lissage temporel et spectral.

    Masque de gain G(f,t) = max(1 - α·Pn/|X|², β) lissé pour
    éviter le bruit musical tout en préservant la parole.
    """
    _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Z) ** 2
    phase = np.angle(Z)

    noise_2d = noise_profile[:, np.newaxis]
    snr = power / (noise_2d + 1e-10)
    gain = np.maximum(1.0 - alpha / (snr + 1e-10), beta)

    if time_smooth > 1:
        gain = uniform_filter1d(gain, size=time_smooth, axis=1)
    if freq_smooth > 1:
        gain = uniform_filter1d(gain, size=freq_smooth, axis=0)
    gain = median_filter(gain, size=(3, 3))
    gain = np.clip(gain, beta, 1.0)

    mag_clean = np.sqrt(power) * gain
    Z_clean = mag_clean * np.exp(1j * phase)
    _, audio_clean = istft(Z_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)

    if len(audio_clean) >= len(audio):
        audio_clean = audio_clean[:len(audio)]
    else:
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))

    return audio_clean