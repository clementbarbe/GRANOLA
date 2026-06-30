"""
Débruitage audio IRM avancé.

1. Estimation adaptative du niveau de bruit (fenêtre glissante)
   La forme spectrale est connue (ref.wav), seule l'amplitude varie.
   On l'estime avec le percentile bas du ratio signal/forme,
   puis minimum glissant + lissage.

2. Filtre de Wiener decision-directed (Ephraim & Malah 1984)
   Le SNR a priori est lissé entre trames successives → transitions
   douces, préservation maximale de la parole.

3. Plancher spectral adapté à la fréquence
   Plus conservateur dans la bande vocale (200-4000 Hz).
"""

import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d, minimum_filter1d

NPERSEG = 1024
NOVERLAP = 768


def estimate_noise_profile(noise_audio, sr, nperseg=NPERSEG, noverlap=NOVERLAP):
    """Puissance moyenne du bruit par bin fréquentiel."""
    _, _, Z = stft(noise_audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return np.mean(np.abs(Z) ** 2, axis=1)


def _adaptive_noise_level(power, noise_shape, sr, adapt_window_s=0.5,
                           percentile=25, nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Estime le facteur d'échelle local du bruit trame par trame.

    Principe : le bruit a une forme spectrale connue (noise_shape).
    Pour chaque trame, on calcule le ratio power / shape par bin.
    Le percentile bas de ce ratio donne le niveau de bruit sans être
    contaminé par la parole (qui ne relève que certains bins).

    Un minimum glissant + lissage suit les variations lentes des gradients.
    """
    # Ratio signal / forme de bruit pour chaque bin et chaque trame
    ratios = power / (noise_shape[:, np.newaxis] + 1e-10)

    # Percentile bas → robuste à la parole
    scale_raw = np.percentile(ratios, percentile, axis=0)

    # Fenêtre en trames
    hop = nperseg - noverlap
    frames_per_sec = sr / hop
    win = max(3, int(adapt_window_s * frames_per_sec))

    # Minimum glissant : suit le plancher de bruit, pas la parole
    scale_min = minimum_filter1d(scale_raw, size=win)
    # Lissage pour éviter les sauts
    scale_smooth = uniform_filter1d(scale_min, size=win * 2)

    # Reconstruire l'estimation de bruit 2D (freq × time)
    noise_est = noise_shape[:, np.newaxis] * scale_smooth[np.newaxis, :]
    return noise_est, scale_smooth


def _decision_directed_wiener(power, noise_est, dd_alpha=0.98):
    """
    Calcule le masque de gain de Wiener avec estimation
    decision-directed du SNR a priori.

    Le gain à la trame t dépend du gain à t-1 → transitions
    très douces, pas de bruit musical, préserve l'attaque des mots.
    """
    n_freq, n_frames = power.shape
    gain = np.zeros_like(power)

    gamma_prev = np.ones(n_freq)

    for t in range(n_frames):
        # SNR a posteriori
        gamma = power[:, t] / (noise_est[:, t] + 1e-10)

        if t == 0:
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            # Decision-directed : mélange du gain précédent et du SNR courant
            xi = (dd_alpha * (gain[:, t - 1] ** 2) * gamma_prev
                  + (1.0 - dd_alpha) * np.maximum(gamma - 1.0, 0.0))

        xi = np.maximum(xi, 1e-6)

        # Gain de Wiener
        gain[:, t] = xi / (xi + 1.0)

        gamma_prev = gamma

    return gain


def _speech_band_floor(freqs, speech_band=(200, 4000),
                       speech_floor=0.10, noise_floor=0.01):
    """
    Plancher de gain qui protège la bande vocale.

    200-4000 Hz → gain minimum 10 %  (on garde les harmoniques)
    En dehors   → gain minimum 1 %   (on supprime agressivement)
    """
    floor = np.full_like(freqs, noise_floor, dtype=float)
    mask = (freqs >= speech_band[0]) & (freqs <= speech_band[1])
    floor[mask] = speech_floor
    # Transition douce aux bords
    floor = uniform_filter1d(floor, size=5)
    return floor


def spectral_subtraction(audio, sr, noise_profile, alpha=1.0, beta=0.05,
                         time_smooth=3, freq_smooth=3,
                         adapt_window_s=0.5,
                         dd_alpha=0.98,
                         speech_band=(200, 4000),
                         speech_floor=0.10,
                         return_extras=False,
                         nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Débruitage complet.

    Paramètres
    ----------
    alpha          : facteur d'échelle sur l'estimation de bruit (1.0 = neutre)
    beta           : plancher global minimum du gain
    adapt_window_s : fenêtre d'adaptation du niveau de bruit (secondes)
    dd_alpha       : lissage decision-directed (0.9-0.99, plus haut = plus doux)
    speech_floor   : plancher dans la bande vocale (0.05-0.15)
    return_extras  : si True, retourne aussi le masque de gain pour les plots
    """
    f_stft, t_stft, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Z) ** 2
    phase = np.angle(Z)
    mag = np.abs(Z)

    # ── 1. Forme spectrale normalisée ──
    noise_shape = noise_profile / (np.sum(noise_profile) + 1e-10)

    # ── 2. Estimation adaptative du niveau ──
    noise_est, noise_scale = _adaptive_noise_level(
        power, noise_shape, sr, adapt_window_s, nperseg=nperseg, noverlap=noverlap
    )
    noise_est *= alpha  # facteur de sursoustraction optionnel

    # ── 3. Gain de Wiener decision-directed ──
    gain = _decision_directed_wiener(power, noise_est, dd_alpha)

    # ── 4. Plancher fréquentiel ──
    floor = _speech_band_floor(f_stft, speech_band, speech_floor, beta)
    gain = np.maximum(gain, floor[:, np.newaxis])

    # ── 5. Lissage du gain ──
    if time_smooth > 1:
        gain = uniform_filter1d(gain, size=time_smooth, axis=1)
    if freq_smooth > 1:
        gain = uniform_filter1d(gain, size=freq_smooth, axis=0)
    gain = np.clip(gain, 0.0, 1.0)

    # ── 6. Reconstruction ──
    Z_clean = mag * gain * np.exp(1j * phase)
    _, audio_clean = istft(Z_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)

    if len(audio_clean) >= len(audio):
        audio_clean = audio_clean[:len(audio)]
    else:
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))

    if return_extras:
        extras = {
            'gain': gain,
            'freqs': f_stft,
            'times': t_stft,
            'noise_scale': noise_scale,
        }
        return audio_clean, extras
    return audio_clean