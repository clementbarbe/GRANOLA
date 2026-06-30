#!/usr/bin/env python3
"""
Pipeline complet : débruitage gradient IRM + transcription Google STT.

Usage
-----
    python main.py --input_dir ./recordings
"""

import argparse
import os
import librosa
import soundfile as sf

from denoise import estimate_noise_profile, spectral_subtraction
from transcribe import transcribe_audio
from plots import (plot_fft_comparison, plot_temporal,
                   plot_periodic_spectrogram, plot_gain_mask)

SR = 16000


def find_wav_files(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    )


def load_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_noise = os.path.join(script_dir, "ref.wav")

    pa = argparse.ArgumentParser(description="Débruitage + STT audio IRM")
    pa.add_argument("--input_dir",    required=True)
    pa.add_argument("--noise_file",   default=default_noise)
    pa.add_argument("--output_dir",   default="./output")
    pa.add_argument("--tr_ms",        type=float, default=1660)
    pa.add_argument("--alpha",        type=float, default=1.0,
                    help="Facteur bruit (1.0=neutre, >1=plus agressif)")
    pa.add_argument("--beta",         type=float, default=0.05,
                    help="Plancher gain hors bande vocale")
    pa.add_argument("--speech_floor", type=float, default=0.10,
                    help="Plancher gain dans bande vocale (200-4000 Hz)")
    pa.add_argument("--dd_alpha",     type=float, default=0.98,
                    help="Lissage decision-directed (0.9-0.99)")
    pa.add_argument("--adapt_window", type=float, default=0.5,
                    help="Fenêtre adaptation bruit (secondes)")
    pa.add_argument("--language",     default="fr-FR")
    pa.add_argument("--no_plots",     action="store_true")
    args = pa.parse_args()

    if not os.path.isfile(args.noise_file):
        print(f"✗ Fichier de bruit introuvable : {args.noise_file}")
        return

    plots_dir = os.path.join(args.output_dir, "plots")
    audio_dir = os.path.join(args.output_dir, "audio_denoised")
    for d in [args.output_dir, plots_dir, audio_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Profil de bruit ──
    print(f"\n{'='*60}")
    print("  PIPELINE AUDIO IRM — WIENER ADAPTATIF + GOOGLE STT")
    print(f"{'='*60}")
    print(f"\n► Bruit de référence : {args.noise_file}")
    noise = load_audio(args.noise_file)
    noise_profile = estimate_noise_profile(noise, SR)
    print(f"  Durée : {len(noise)/SR:.1f}s")

    # ── 2. Fichiers ──
    wav_files = find_wav_files(args.input_dir)
    noise_abs = os.path.abspath(args.noise_file)
    wav_files = [f for f in wav_files if os.path.abspath(f) != noise_abs]
    print(f"► {len(wav_files)} fichier(s) à traiter\n")

    if not wav_files:
        print("Aucun .wav trouvé.")
        return

    # ── 3. Traitement ──
    results = []

    for i, path in enumerate(wav_files):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"{'—'*60}")
        print(f"[{i+1}/{len(wav_files)}]  {name}")

        raw = load_audio(path)
        print(f"  Durée : {len(raw)/SR:.1f}s")

        # Débruitage avec extras pour les plots
        clean, extras = spectral_subtraction(
            raw, SR, noise_profile,
            alpha=args.alpha,
            beta=args.beta,
            speech_floor=args.speech_floor,
            dd_alpha=args.dd_alpha,
            adapt_window_s=args.adapt_window,
            return_extras=True,
        )

        out_wav = os.path.join(audio_dir, f"{name}_denoised.wav")
        sf.write(out_wav, clean, SR)
        print(f"  ✓ Débruité → {out_wav}")

        # Plots
        if not args.no_plots:
            plot_fft_comparison(raw, clean, SR, title=name,
                save_path=os.path.join(plots_dir, f"{name}_fft.png"))
            plot_temporal(raw, clean, SR, title=name,
                save_path=os.path.join(plots_dir, f"{name}_temporal.png"))
            plot_periodic_spectrogram(raw, clean, SR, tr_ms=args.tr_ms,
                title=name,
                save_path=os.path.join(plots_dir, f"{name}_periodic.png"))
            plot_gain_mask(extras, SR, title=name,
                save_path=os.path.join(plots_dir, f"{name}_gain_mask.png"))
            print(f"  ✓ Plots → {plots_dir}/")

        # Transcription
        print("  ⏳ Transcription (brut)…")
        text_raw = transcribe_audio(raw, SR, args.language)
        print("  ⏳ Transcription (débruité)…")
        text_clean = transcribe_audio(clean, SR, args.language)

        print(f"  ╔ BRUT     : {text_raw or '(vide)'}")
        print(f"  ╚ DÉBRUITÉ : {text_clean or '(vide)'}")

        results.append(dict(file=name, raw=text_raw, clean=text_clean))

    # ── 4. Résultats ──
    out_txt = os.path.join(args.output_dir, "transcriptions.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPTIONS AUDIO IRM — WIENER ADAPTATIF\n")
        f.write(f"{'='*60}\n")
        f.write(f"Langue       : {args.language}\n")
        f.write(f"Alpha        : {args.alpha}\n")
        f.write(f"Beta         : {args.beta}\n")
        f.write(f"Speech floor : {args.speech_floor}\n")
        f.write(f"DD alpha     : {args.dd_alpha}\n")
        f.write(f"Adapt window : {args.adapt_window}s\n")
        f.write(f"TR           : {args.tr_ms} ms\n")
        f.write(f"{'='*60}\n\n")

        all_raw, all_clean = [], []

        for r in results:
            w_raw = r["raw"].lower().split()
            w_cln = r["clean"].lower().split()
            all_raw.extend(w_raw)
            all_clean.extend(w_cln)

            f.write(f"--- {r['file']} ---\n")
            f.write(f"  BRUT     : {r['raw']}\n")
            f.write(f"  DÉBRUITÉ : {r['clean']}\n")
            f.write(f"  Mots bruts    : {w_raw}\n")
            f.write(f"  Mots débruités: {w_cln}\n\n")

        f.write(f"\n{'='*60}\n")
        f.write("MOTS — SANS DÉBRUITAGE\n")
        f.write(f"{'='*60}\n")
        f.write("\n".join(all_raw) + "\n")

        f.write(f"\n{'='*60}\n")
        f.write("MOTS — AVEC DÉBRUITAGE\n")
        f.write(f"{'='*60}\n")
        f.write("\n".join(all_clean) + "\n")

    print(f"\n{'='*60}")
    print(f"  ✓ Transcriptions : {out_txt}")
    print(f"  ✓ Audio débruité : {audio_dir}/")
    if not args.no_plots:
        print(f"  ✓ Plots          : {plots_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()