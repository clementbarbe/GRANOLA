"""Visualisations : FFT, temporel, spectrogramme périodique, masque de gain."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter1d

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ──────────────────────────────────────────────
#  1. FFT avant / après
# ──────────────────────────────────────────────

def plot_fft_comparison(audio_raw, audio_clean, sr, title="", save_path=None):
    n = min(len(audio_raw), len(audio_clean))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    db_raw = 20 * np.log10(np.abs(np.fft.rfft(audio_raw[:n])) + 1e-10)
    db_clean = 20 * np.log10(np.abs(np.fft.rfft(audio_clean[:n])) + 1e-10)

    k = max(1, len(freqs) // 500)
    db_raw = uniform_filter1d(db_raw, k)
    db_clean = uniform_filter1d(db_clean, k)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(freqs, db_raw, alpha=0.8, label="Avant débruitage", color="#e74c3c")
    ax.plot(freqs, db_clean, alpha=0.8, label="Après débruitage", color="#2ecc71")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Comparaison FFT — {title}")
    ax.legend()
    ax.set_xlim(0, sr / 2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
#  2. Signal temporel
# ──────────────────────────────────────────────

def plot_temporal(audio_raw, audio_clean, sr, title="", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    t_raw = np.arange(len(audio_raw)) / sr
    t_clean = np.arange(len(audio_clean)) / sr

    axes[0].plot(t_raw, audio_raw, lw=0.3, color="#e74c3c")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Signal brut — {title}")

    axes[1].plot(t_clean, audio_clean, lw=0.3, color="#2ecc71")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Temps (s)")
    axes[1].set_title(f"Signal débruité — {title}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
#  3. Spectrogramme périodique (pliage au TR)
# ──────────────────────────────────────────────

def plot_periodic_spectrogram(audio_raw, audio_clean, sr, tr_ms=1660,
                              title="", save_path=None):
    tr_samp = int(tr_ms / 1000.0 * sr)
    n = min(len(audio_raw), len(audio_clean))
    n_trs = n // tr_samp

    if n_trs < 2:
        print(f"  ⚠ Signal trop court pour l'analyse périodique")
        return

    nps, nov = 256, 192

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, (audio, label) in enumerate(
        [(audio_raw[:n], "Brut"), (audio_clean[:n], "Débruité")]
    ):
        f_f, t_f, Sxx_f = spectrogram(audio, fs=sr, nperseg=nps, noverlap=nov)
        axes[0, col].pcolormesh(
            t_f, f_f, 10 * np.log10(Sxx_f + 1e-10),
            shading="gouraud", cmap="inferno",
        )
        for k in range(1, n_trs):
            axes[0, col].axvline(k * tr_ms / 1000, color="cyan", lw=0.4, alpha=0.5)
        axes[0, col].set_title(f"Spectrogramme {label}")
        axes[0, col].set_ylabel("Fréquence (Hz)")
        axes[0, col].set_ylim(0, sr / 2)

        segments = audio[: n_trs * tr_samp].reshape(n_trs, tr_samp)
        Sxx_avg = None
        for seg in segments:
            f_t, t_t, Sxx_s = spectrogram(seg, fs=sr, nperseg=nps, noverlap=nov)
            Sxx_avg = Sxx_s if Sxx_avg is None else Sxx_avg + Sxx_s
        Sxx_avg /= n_trs

        im = axes[1, col].pcolormesh(
            t_t * 1000, f_t, 10 * np.log10(Sxx_avg + 1e-10),
            shading="gouraud", cmap="inferno",
        )
        axes[1, col].set_title(f"Moyen / TR ({label}, n={n_trs})")
        axes[1, col].set_ylabel("Fréquence (Hz)")
        axes[1, col].set_xlabel("Temps dans le TR (ms)")
        axes[1, col].set_ylim(0, sr / 2)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Puissance (dB)",
                 shrink=0.5, pad=0.02)
    fig.suptitle(f"Analyse périodique EPI (TR = {tr_ms} ms) — {title}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
#  4. Masque de gain + courbe d'adaptation
# ──────────────────────────────────────────────

def plot_gain_mask(extras, sr, title="", save_path=None):
    """
    Affiche :
    - Haut : le masque de gain (freq × time), vert = conservé, rouge = supprimé
    - Bas  : le facteur d'échelle du bruit estimé dans le temps
    """
    gain = extras['gain']
    freqs = extras['freqs']
    times = extras['times']
    noise_scale = extras['noise_scale']

    fig, axes = plt.subplots(2, 1, figsize=(15, 8),
                             gridspec_kw={'height_ratios': [3, 1]})

    # ── Masque de gain ──
    im = axes[0].pcolormesh(
        times, freqs, gain,
        shading='gouraud', cmap='RdYlGn', vmin=0, vmax=1,
    )
    axes[0].set_ylabel("Fréquence (Hz)")
    axes[0].set_title(f"Masque de gain Wiener — {title}")
    axes[0].set_ylim(0, sr / 2)
    fig.colorbar(im, ax=axes[0], label="Gain (0 = supprimé, 1 = conservé)",
                 shrink=0.8)

    # ── Bandes vocales repères ──
    for f in [200, 4000]:
        axes[0].axhline(f, color='white', ls='--', lw=0.8, alpha=0.7)
    axes[0].text(times[5], 2200, "bande vocale",
                 color='white', fontsize=9, fontstyle='italic')

    # ── Facteur d'échelle du bruit ──
    axes[1].plot(times[:len(noise_scale)], noise_scale,
                 color='#3498db', lw=1)
    axes[1].fill_between(times[:len(noise_scale)], noise_scale,
                         alpha=0.2, color='#3498db')
    axes[1].set_xlabel("Temps (s)")
    axes[1].set_ylabel("Niveau de bruit\n(échelle relative)")
    axes[1].set_title("Estimation adaptative du niveau de bruit")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()