#!/usr/bin/env python3
"""
GRANOLA — GRAdient NOise Less Audio
Pipeline : débruitage gradient IRM + transcription Google STT.
"""

import argparse
import os
import librosa
import soundfile as sf

from denoise import (estimate_noise_profile, spectral_subtraction,
                     DEFAULT_ALPHA, DEFAULT_BETA,
                     DEFAULT_TIME_SMOOTH, DEFAULT_FREQ_SMOOTH)
from transcribe import transcribe_audio
from plots import plot_fft_comparison, plot_temporal, plot_periodic_spectrogram

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

    pa = argparse.ArgumentParser(
        description="GRANOLA — GRAdient NOise Less Audio")
    pa.add_argument("--input_dir",  required=True, help="Dossier des .wav")
    pa.add_argument("--noise_file", default=default_noise,
                    help="Bruit gradient pur (défaut : ref.wav)")
    pa.add_argument("--output_dir", default="./output")
    pa.add_argument("--tr_ms",      type=float, default=1660)
    pa.add_argument("--alpha",      type=float, default=DEFAULT_ALPHA)
    pa.add_argument("--beta",       type=float, default=DEFAULT_BETA)
    pa.add_argument("--time_smooth", type=int,  default=DEFAULT_TIME_SMOOTH)
    pa.add_argument("--freq_smooth", type=int,  default=DEFAULT_FREQ_SMOOTH)
    pa.add_argument("--language",   default="fr-FR")
    pa.add_argument("--no_plots",   action="store_true")
    args = pa.parse_args()

    if not os.path.isfile(args.noise_file):
        print(f"✗ Fichier de bruit introuvable : {args.noise_file}")
        print("  Place ref.wav à côté de main.py ou utilise --noise_file")
        return

    plots_dir = os.path.join(args.output_dir, "plots")
    audio_dir = os.path.join(args.output_dir, "audio_denoised")
    for d in [args.output_dir, plots_dir, audio_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*60}")
    print("  🥣 GRANOLA — GRAdient NOise Less Audio")
    print(f"{'='*60}")
    print(f"\n► Bruit de référence : {args.noise_file}")
    noise = load_audio(args.noise_file)
    noise_profile = estimate_noise_profile(noise, SR)
    print(f"  Durée : {len(noise)/SR:.1f}s")
    print(f"► Paramètres : α={args.alpha}  β={args.beta}  "
          f"ts={args.time_smooth}  fs={args.freq_smooth}")

    wav_files = find_wav_files(args.input_dir)
    noise_abs = os.path.abspath(args.noise_file)
    wav_files = [f for f in wav_files if os.path.abspath(f) != noise_abs]
    print(f"► {len(wav_files)} fichier(s) à traiter\n")

    if not wav_files:
        print("Aucun .wav trouvé.")
        return

    results = []

    for i, path in enumerate(wav_files):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"{'—'*60}")
        print(f"[{i+1}/{len(wav_files)}]  {name}")

        raw = load_audio(path)
        print(f"  Durée : {len(raw)/SR:.1f}s")

        clean = spectral_subtraction(
            raw, SR, noise_profile,
            alpha=args.alpha, beta=args.beta,
            time_smooth=args.time_smooth, freq_smooth=args.freq_smooth,
        )
        out_wav = os.path.join(audio_dir, f"{name}_denoised.wav")
        sf.write(out_wav, clean, SR)
        print(f"  ✓ Débruité → {out_wav}")

        if not args.no_plots:
            plot_fft_comparison(raw, clean, SR, title=name,
                save_path=os.path.join(plots_dir, f"{name}_fft.png"))
            plot_temporal(raw, clean, SR, title=name,
                save_path=os.path.join(plots_dir, f"{name}_temporal.png"))
            plot_periodic_spectrogram(raw, clean, SR, tr_ms=args.tr_ms,
                title=name,
                save_path=os.path.join(plots_dir, f"{name}_periodic.png"))
            print(f"  ✓ Plots → {plots_dir}/")

        print("  ⏳ Transcription (brut)…")
        text_raw = transcribe_audio(raw, SR, args.language)
        print("  ⏳ Transcription (débruité)…")
        text_clean = transcribe_audio(clean, SR, args.language)

        print(f"  ╔ BRUT     : {text_raw or '(vide)'}")
        print(f"  ╚ DÉBRUITÉ : {text_clean or '(vide)'}")

        results.append(dict(file=name, raw=text_raw, clean=text_clean))

    out_txt = os.path.join(args.output_dir, "transcriptions.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("🥣 GRANOLA — GRAdient NOise Less Audio\n")
        f.write(f"{'='*60}\n")
        f.write(f"Langue : {args.language}\n")
        f.write(f"Alpha  : {args.alpha}  |  Beta : {args.beta}\n")
        f.write(f"Time smooth : {args.time_smooth}  |  Freq smooth : {args.freq_smooth}\n")
        f.write(f"TR     : {args.tr_ms} ms\n")
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