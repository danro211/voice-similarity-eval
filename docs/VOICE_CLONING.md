# Synthetic Voice Generation Guide (Zero-Shot)

This guide details the technical process used in this project to clone vocal identities from short audio samples, utilizing the open-source foundational model **XTTS v2**.

## Theoretical Background: XTTS v2 Model

For the voice similarity tests, a *Zero-Shot* cloning approach was chosen. This means the Artificial Intelligence model does not require exhaustive training (fine-tuning) or hours of studio recordings to mimic a new speaker. 

By providing a single reference audio of 3 to 5 seconds, the XTTS v2 neural network extracts the individual's "acoustic footprint" (pitch, timbre, and cadence) and conditions it on an input text, generating a highly realistic synthetic voice with native Spanish support.

## Execution Environment

Due to strict dependency version incompatibilities between the evaluation libraries and the synthesis engine, the voice generation must be executed in an isolated Conda environment:

```bash
# Activate the generation environment
conda activate generador-tts
```

## Workflow

### 1. Reference Pre-processing and Normalization
The system (both XTTS v2 and the subsequent biometric classifier) requires the base audio to meet strict standards: **`.wav` format, 16 kHz sampling rate, and a single audio channel (mono)**.

If a voice note is obtained from messaging platforms (usually in `.m4a` or `.ogg` format), it is mandatory to normalize it using `ffmpeg` before feeding it into the pipeline. Run the following command from the root of the project to perform the conversion and save the result in the ingestion folder:

```bash
ffmpeg -i path/to/original_audio.m4a -ac 1 -ar 16000 data/raw/real_name_reference.wav
```

### 2. Clone Generation
Once the reference audio is normalized in `data/raw/`, the generation script will process the cloning. Execute the following command:

```bash
python scripts/generacion/generar_sintetico.py
```

### 3. Data Output and Next Steps
The XTTS v2 inference process is highly GPU-intensive (CUDA acceleration). Upon completion, the system will generate the synthetic audio and automatically save it in the **`data/raw/`** directory (e.g., `data/raw/synthetic_name_xtts.wav`). 

By placing the output in this same folder, the cloned audio is standardized and ready for the main evaluation pipeline (`make all` from the `ise-protocol` environment) to detect it and automatically calculate the Cosine Similarity, MCD distances, and NISQA predictions.
