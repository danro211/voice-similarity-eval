# Ethical and Legal Considerations

This project handles sensitive biometric data, specifically human voice recordings and extracted speaker embeddings (d-vectors). As outlined in the technical documentation of this project, the following ethical and legal guidelines are strictly enforced:

## 1. Informed Consent and Privacy
All human audio samples collected and stored in the `data/` directory were obtained with the explicit, informed consent of the participants. The extracted biometric representations (d-vectors) are treated as confidential data and are used exclusively for academic research purposes at the Universidad de Costa Rica (UCR). 

## 2. Dual-Use Technology and Deepfakes
While this repository utilizes Voice Synthesis and Cloning (XTTS v2) technologies, its primary objective is analytical and defensive. The framework is designed to establish reproducible biometric baselines that help evaluate TTS systems and distinguish between legitimate human voices and synthetic spoofing attacks (deepfakes). The authors strictly prohibit the use of this pipeline for identity theft, extortion, fraud, or disinformation.

## 3. Transparency and Disclosure
Any synthetic voice generated using this framework is explicitly labeled as synthetic (e.g., `sintetico`) in the dataset and outputs. In alignment with ethical AI practices, we emphasize that any real-world application derived from this protocol must transparently disclose to users when they are interacting with synthetic speech.

## 4. Academic License and Data Usage
The source code provided in the `scripts/` directory is provided for academic and educational use. However, the human audio datasets included in this repository are strictly for reproducing the experiments documented in this project. They may not be extracted, redistributed, or used to train commercial AI models without explicit, written permission from the individuals involved.
