#!/usr/bin/env python3
"""
GRANOLA — GRAdient NOise Less Audio
Pipeline : débruitage gradient IRM + transcription Google STT + timecodes.
"""

import argparse
import os
import librosa
import soundfile as sf

from denoise import (estimate_noise_profile, spectral_subtraction,
                     DEFAULT_ALPHA, DEFAULT_BETA,
                     DEFAULT_TIME_SMOOTH, DEFAULT_FREQ_SMOOTH)
from transcribe import transcribe_google, detect_speech_segments, match_words_to_segments
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


def write_tsv(path, words):
    with open(path, "w", encoding="utf-8") as f:
        f.write("start\tend\tword\n")
        for w in words:
            f.write(f"{w['start']:.3f}\t{w['end']:.3f}\t{w['word']}\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_noise = os.path.join(script_dir, "ref.wav")

    pa = argparse.ArgumentParser(
        description="GRANOLA — GRAdient NOise Less Audio")
    pa.add_argument("--input_dir",   required=True)
    pa.add_argument("--noise_file",  default=default_noise)
    pa.add_argument("--output_dir",  default="./output")
    pa.add_argument("--tr_ms",       type=float, default=1660)
    pa.add_argument("--alpha",       type=float, default=DEFAULT_ALPHA)
    pa.add_argument("--beta",        type=float, default=DEFAULT_BETA)
    pa.add_argument("--time_smooth", type=int,   default=DEFAULT_TIME_SMOOTH)
    pa.add_argument("--freq_smooth", type=int,   default=DEFAULT_FREQ_SMOOTH)
    pa.add_argument("--language",    default="fr-FR")
    pa.add_argument("--stt_source",  choices=["raw", "denoised"], default="raw",
                    help="Signal utilisé pour le STT (défaut : raw)")
    pa.add_argument("--plots",       action="store_true")
    pa.add_argument("--compare",     action="store_true")
    args = pa.parse_args()

    if not os.path.isfile(args.noise_file):
        print(f"✗ Fichier de bruit introuvable : {args.noise_file}")
        return

    tsv_dir = os.path.join(args.output_dir, "transcriptions")
    audio_dir = os.path.join(args.output_dir, "audio_denoised")
    plots_dir = os.path.join(args.output_dir, "plots")
    for d in [args.output_dir, tsv_dir, audio_dir]:
        os.makedirs(d, exist_ok=True)
    if args.plots:
        os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  🥣 GRANOLA — GRAdient NOise Less Audio")
    print(f"{'='*60}")

    noise = load_audio(args.noise_file)
    noise_profile = estimate_noise_profile(noise, SR)
    print(f"► Bruit      : {args.noise_file} ({len(noise)/SR:.1f}s)")
    print(f"► Paramètres : α={args.alpha}  β={args.beta}  "
          f"ts={args.time_smooth}  fs={args.freq_smooth}")
    print(f"► STT sur    : {args.stt_source}")

    wav_files = find_wav_files(args.input_dir)
    noise_abs = os.path.abspath(args.noise_file)
    wav_files = [f for f in wav_files if os.path.abspath(f) != noise_abs]
    print(f"► {len(wav_files)} fichier(s)\n")

    if not wav_files:
        print("Aucun .wav trouvé.")
        return

    for i, path in enumerate(wav_files):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"{'—'*60}")
        print(f"[{i+1}/{len(wav_files)}]  {name}")

        raw = load_audio(path)

        # Débruitage
        clean = spectral_subtraction(
            raw, SR, noise_profile,
            alpha=args.alpha, beta=args.beta,
            time_smooth=args.time_smooth, freq_smooth=args.freq_smooth,
        )
        out_wav = os.path.join(audio_dir, f"{name}_denoised.wav")
        sf.write(out_wav, clean, SR)
        print(f"  ✓ Débruité")

        # Plots
        if args.plots:
            plot_fft_comparison(raw, clean, SR, title=name,
                save_path=os.path.join(plots_dir, f"{name}_fft.png"))
            plot_temporal(raw, clean, SR, title=name,
                save_path=os.path.join(plots_dir, f"{name}_temporal.png"))
            plot_periodic_spectrogram(raw, clean, SR, tr_ms=args.tr_ms,
                title=name,
                save_path=os.path.join(plots_dir, f"{name}_periodic.png"))

        # ── Choix du signal pour le STT ──
        stt_audio = raw if args.stt_source == "raw" else clean

        # ── Timecodes toujours sur le débruité (meilleure détection) ──
        segments = detect_speech_segments(clean, SR)

        # ── Comparaison optionnelle ──
        if args.compare:
            print("  ⏳ STT brut…")
            words_r = transcribe_google(raw, SR, args.language)
            segs_r = detect_speech_segments(raw, SR)
            matched_r = match_words_to_segments(words_r, segs_r)
            write_tsv(os.path.join(tsv_dir, f"{name}_brut.tsv"), matched_r)
            print(f"  BRUT     : {' '.join(words_r) or '(vide)'}")

            print("  ⏳ STT débruité…")
            words_d = transcribe_google(clean, SR, args.language)
            matched_d = match_words_to_segments(words_d, segments)
            write_tsv(os.path.join(tsv_dir, f"{name}_denoised.tsv"), matched_d)
            print(f"  DÉBRUITÉ : {' '.join(words_d) or '(vide)'}")

        # ── Transcription principale ──
        print(f"  ⏳ STT ({args.stt_source})…")
        words = transcribe_google(stt_audio, SR, args.language)
        matched = match_words_to_segments(words, segments)
        write_tsv(os.path.join(tsv_dir, f"{name}.tsv"), matched)
        print(f"  RÉSULTAT : {' '.join(words) or '(vide)'}")

        for w in matched[:5]:
            print(f"    [{w['start']:6.2f}s → {w['end']:6.2f}s]  {w['word']}")
        if len(matched) > 5:
            print(f"    ... +{len(matched) - 5} mots")

    print(f"\n{'='*60}")
    print(f"  ✓ TSV   → {tsv_dir}/")
    print(f"  ✓ Audio → {audio_dir}/")
    if args.plots:
        print(f"  ✓ Plots → {plots_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()