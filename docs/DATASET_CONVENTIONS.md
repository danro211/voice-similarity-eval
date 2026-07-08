# Dataset Naming Conventions

This document outlines the strict naming conventions required for the audio files in the `data/raw/` directory. 

The automated evaluation pipeline (specifically the EER calculation and the Cosine Similarity matrices) parses the filenames to dynamically label comparisons as either "genuine" (same speaker) or "imposter/clone" (different speaker or synthetic attack).

## Required File Naming Structure

All audio files must adhere to the following structure:
`[speaker_name]_[type]_[identifier].wav`

### 1. Natural / Real Voices (References)
For genuine human audio recordings, the `[type]` must be explicitly labeled as `real`.
* **Format:** `[name]_real_[number].wav`
* **Examples:** * `daniel_real_1.wav`
  * `felipe_real_2.wav`
  * `paula_real_1.wav`

### 2. Synthetic / Cloned Voices
For audios generated via XTTS v2 or any other TTS engine, the `[type]` must be explicitly labeled as `sintetico`.
* **Format:** `[name]_sintetico_[engine/identifier].wav`
* **Examples:**
  * `daniel_sintetico_xtts.wav`
  * `marvin_sintetico_1_forms.wav`

## Audio Formatting Requirements
Regardless of the naming convention, the system's underlying models (ECAPA-TDNN and NISQA) require the audio files to be strictly formatted as:
* **Format:** `.wav`
* **Sample Rate:** 16,000 Hz (16 kHz)
* **Channels:** 1 (Mono)

Failure to follow this naming and formatting convention will result in parsing errors during the `make all` execution.
