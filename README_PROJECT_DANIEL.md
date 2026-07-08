# Electrical Project: Voice Similarity Evaluation Protocol

This repository contains the technical implementation for the project "Design and Implementation of a Reproducible Voice Similarity Evaluation Protocol for TTS and Voice Conversion Systems," developed at the School of Electrical Engineering, Universidad de Costa Rica (UCR).

The core objective of this project is to provide an automated framework to evaluate the fidelity of cloned voices against their original speakers by cross-referencing objective computational metrics with human perceptual assessments.

## Repository Structure

The project architecture is designed to logically separate raw data, execution scripts, and generated artifacts to ensure full reproducibility:

* **`data/`**: Contains the audio files in `.wav` format (16 kHz, mono).
  * `raw/`: Original reference audios and voice notes.
  * `processed/`: Pre-processed audios and generated synthetic clones.
* **`docs/`**: Formal project documentation, including logbooks (bitácoras), technical PDF reports, and specific guides.
* **`metrics/`**: Submodules and dependencies for specific objective metrics (e.g., the NISQA model).
* **`models/`**: Locally downloaded weights for the neural network models.
* **`outputs/`**: Target directory for all generated artifacts.
  * `graphs/`: ROC curves, cosine similarity heatmaps, and MOS bar charts.
  * `reports/`: `.csv` files containing the calculated distance results.
* **`scripts/`**: Main source code for the evaluation pipeline.
  * `generacion/`: Scripts for voice synthesis and cloning.
  * `evaluacion/`: Scripts to calculate Cosine Similarity, MCD, and NISQA scores.
  * `utils/`: Helper functions and data processing tools.
* **`Makefile`**: Orchestration file to execute the entire workflow with a single command.
* **`requirements.txt`**: List of required Python dependencies.

## Requirements and Development Environment

To guarantee reproducibility and prevent dependency conflicts between the Deep Learning libraries and the synthesis engine, this project utilizes isolated virtual environments managed by Anaconda (Conda).

**Main Dependencies:**
* Python 3.9+
* `torch`, `torchaudio` (CUDA support required for GPU acceleration, optimized for RTX 50-series or similar)
* `speechbrain` (for d-vector extraction via ECAPA-TDNN)
* `librosa` (for spectral analysis and MCD calculation)
* `pandas`, `matplotlib`, `seaborn` (for data analysis and plotting)
* `ffmpeg` (system tool for audio standardization)

### Setting Up the Base Environment (Evaluation)
This environment is used to run the entire analytical pipeline:

```bash
conda create -n ise-protocol python=3.10
conda activate ise-protocol
pip install -r requirements.txt
```

*(Note: A separate environment is required for generating the synthetic voices. Please refer to `docs/CLONACION_VOCES.md` for detailed instructions).*

## Usage and Pipeline Execution

The project is fully orchestrated via the `Makefile`. To run the complete analysis on the audio files located in `data/raw/`, simply execute the following commands from the root directory:

```bash
# Activate the evaluation environment
conda activate ise-protocol

# Execute the complete protocol (Extraction, Distances, MCD, and Plotting)
make all
```

The final results, including the comparative graphs and the Cosine Similarity Matrix, will be automatically generated and saved inside the `outputs/` folder.

## Authorship and Credits

* **Author:** Daniel Valverde Ortiz
* **Academic Advisor:** Ing. Marvin Coto Jiménez, PhD.
* **Institution:** Universidad de Costa Rica (UCR) - LAPA-CR Laboratory
