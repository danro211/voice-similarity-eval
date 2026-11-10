# voice-similarity-eval

**A reproducible evaluation toolkit for voice similarity, speaker identity preservation, and perceptual quality in TTS and voice conversion systems.**

This repository provides a **modular, low-cost, and ethically grounded** framework to evaluate **how similar a synthetic voice is to a natural reference voice**, with a strong focus on **assistive speech technologies** and **Spanish (including Costa Rican variants)**.

It is designed for:

* academic research,
* student-driven projects,
* evaluation of TTS, voice cloning, and voice conversion systems,
* responsible development of personalized artificial voices.

---

## Project goals

1. Evaluate **speaker similarity** between reference and synthetic speech.
2. Compare **objective similarity metrics** with **human perceptual judgments**.
3. Study **robustness** under noise, compression, and channel effects.
4. Provide **ethical and reproducible evaluation protocols** for assistive voices.
5. Enable **large-scale student participation** through modular subprojects.

---

## What does this repository evaluate?

| Dimension                       | Description                                               |
| ------------------------------- | --------------------------------------------------------- |
| **Speaker identity similarity** | Does the synthetic voice sound like the same person?      |
| **Perceptual quality**          | Is the voice natural, pleasant, and intelligible?         |
| **Consistency**                 | Is similarity preserved across different utterances?      |
| **Robustness**                  | How does similarity degrade with noise or compression?    |
| **Age / style cues**            | Does the perceived age or speaking style remain coherent? |

---

## Repository structure

```text
voice-similarity-eval/
├── data/
│   ├── reference/        # natural reference voices
│   ├── synthesized/     # TTS / VC outputs
│   ├── metadata.csv     # speaker, age, language, conditions
│   └── README.md        # data preparation guidelines
│
├── features/
│   ├── embeddings.py    # speaker embedding extraction
│   ├── melspec.py       # mel-spectrograms
│   └── prosody.py       # pitch, energy, speaking rate
│
├── metrics/
│   ├── cosine_similarity.py
│   ├── speaker_verification.py
│   ├── mcd.py           # Mel Cepstral Distortion
│   └── summary.py
│
├── perceptual_tests/
│   ├── mos/
│   │   ├── forms/
│   │   └── analysis.py
│   ├── abx/
│   │   ├── generation.py
│   │   └── analysis.py
│   └── README.md
│
├── robustness/
│   ├── noise.py
│   ├── compression.py
│   └── replay.py
│
├── notebooks/
│   ├── baseline_similarity.ipynb
│   ├── mos_analysis.ipynb
│   └── robustness_study.ipynb
│
├── ethics/
│   ├── consent_templates.md
│   ├── usage_policy.md
│   └── risk_analysis.md
│
├── results/
│   └── examples/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Implemented evaluation metrics (baseline)

### Objective metrics

* **Cosine similarity** between speaker embeddings
* **Speaker verification scores** (same / different speaker)
* **Mel Cepstral Distortion (MCD)**
* Statistical distances in pitch, energy, and duration

> ⚠️ Important: Objective metrics **do not replace human perception**.
> This toolkit is designed to **compare and study correlations**, not to replace listening tests.

---

### Human perceptual evaluation

#### MOS (Mean Opinion Score)

* Naturalness
* Speaker similarity
* Clarity / intelligibility

#### ABX tests

* **A**: reference voice
* **B**: synthetic voice 1
* **X**: synthetic voice 2

Question: *Which voice (B or X) sounds more similar to A?*

The repository includes:

* automatic pair generation,
* test forms,
* statistical analysis scripts.

---

## Recommended evaluation protocols

1. **Clean condition**

   * Reference vs. synthetic speech
2. **Noise robustness**

   * Controlled SNR levels
3. **Compression robustness**

   * WhatsApp / Opus / MP3
4. **Cross-utterance consistency**

   * Same speaker, different sentences
5. **Longitudinal consistency**

   * Same voice across multiple sessions

---

## Typical use cases

* Comparing multiple TTS systems for the same target voice
* Evaluating identity preservation in assistive speech systems
* Measuring degradation through real communication channels
* Course projects in audio, ML, and AI
* Perceptual studies with student listeners

---

## Student-friendly research projects

* Implement new similarity metrics
* Compare different speaker embedding models
* Analyze bias related to perceived age or gender
* Improve ABX experimental design
* Study correlation between objective metrics and MOS
* Extend the toolkit to Costa Rican Spanish

Each topic can be developed **independently**, making the repository ideal for large cohorts.

---

## Ethics and responsible use

This repository **is not intended for identity impersonation**.

Mandatory principles:

* Informed consent from all recorded speakers
* Research-only and assistive use cases
* No release of models trained on private voices
* Transparent reporting of errors and limitations

See:

* `ethics/consent_templates.md`
* `ethics/usage_policy.md`

---

## Quick installation

```bash
git clone https://github.com/ORG/voice-similarity-eval.git
cd voice-similarity-eval
pip install -r requirements.txt
```

---

## Minimal example

```python
from metrics.cosine_similarity import compute_similarity

score = compute_similarity(
    reference_wav="data/reference/ref_01.wav",
    synthetic_wav="data/synthesized/syn_01.wav"
)

print("Similarity score:", score)
```

---

## Contributing

We welcome issues and pull requests! If you are a student at Universidad de Costa Rica, please check the "Good First Issue" tab to find tasks suitable for beginners.

Contact: Marvin Coto - marvin.coto@ucr.ac.cr

---

## License

**MIT License**, except for audio files, which must follow the specific licenses described in `data/README.md`.

---

## Final note

> *Voice similarity is not an absolute truth — it is perceptual, contextual, and ethical.*

This project aims to **measure it rigorously, transparently, and responsibly**.

* or create **starter GitHub issues** for student contributors.

---

## Estudio de similitud vocal entre hermanos (IE-0499)

Esta extension investiga la similitud vocal natural entre hermanos mediante
representaciones de hablante ECAPA-TDNN y caracteristicas acusticas.
Se comparan voces naturales, no voces clonadas.

**Procesamiento:** `data/family_study/` contiene el protocolo, las
comparaciones acusticas y de embeddings, la integracion y la auditoria.
La extraccion de embeddings se encuentra en
`scripts/evaluacion/extraer_embeddings_familia.py`.

**Visualizaciones:** los generadores, componentes compartidos y pruebas
estan organizados en `data/family_study/visualizations/`.
Consultar su [documentacion](data/family_study/visualizations/README.md).

**Privacidad:** esta ampliacion incorpora codigo, pero no nuevas
grabaciones familiares, identificadores de participantes, embeddings ni
resultados individuales. El repositorio original ya contiene archivos
WAV de trabajos anteriores en `data/raw/`; no corresponden a este
conjunto de datos familiar. La reproduccion completa requiere acceso
autorizado a los datos y a las dependencias del proyecto.

**Verificaciones realizadas:** seis pruebas automatizadas con datos
sinteticos y tres verificaciones de generadores con resultados reales.
Estas pruebas no representan una reproduccion integral desde cero.
