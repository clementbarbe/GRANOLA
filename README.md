# 🥣 GRANOLA — GRAdient NOise Less Audio

Pipeline Python de débruitage et transcription d'enregistrements vocaux
acquis dans un scanner IRM (séquences EPI).

## Principe

Le bruit de gradient IRM est quasi-stationnaire sur la durée d'un run.
GRANOLA utilise un enregistrement de **bruit pur** (`ref.wav`) pour
construire un profil spectral de référence, puis applique un **filtre de
Wiener lissé** pour supprimer le bruit tout en préservant la parole.

Les mots sont ensuite transcrits via **Google Web Speech API** (gratuit,
sans clé, nécessite une connexion internet).

## Installation

```bash
pip install numpy scipy matplotlib librosa soundfile SpeechRecognition
```

## Structure

```csharp
GRANOLA/
├── main.py              # Pipeline principal
├── denoise.py           # Filtre de Wiener lissé
├── transcribe.py        # Transcription Google STT
├── plots.py             # Visualisations
├── compare_params.py    # Recherche de paramètres optimaux
├── ref.wav              # Enregistrement de bruit gradient pur
└── README.md
```
## Utilisation rapide

Placer ref.wav (bruit gradient pur) à la racine du projet, puis :

```bash
python main.py --input_dir ./data
```

## Sortie

```csharp
output/
├── transcriptions.txt        # Mots détectés (brut vs débruité)
├── audio_denoised/            # Fichiers .wav débruités
└── plots/                     # FFT, temporel, spectrogramme EPI
```

## Paramètres

Paramètre	Défaut	Description
--tr_ms	1660	TR de la séquence EPI (pour les plots)
--language	fr-FR	Langue de transcription
--no_plots	—	Désactiver les plots

```bash
# Exemples
python main.py --input_dir ./data --alpha 1.0
python main.py --input_dir ./data --no_plots
python main.py --input_dir ./data --language en-US
```


## Contexte

Développé pour le débruitage d'enregistrements vocaux acquis avec un
microphone à débruitage actif dans une IRM 3T (séquences EPI, TR = 1660 ms).