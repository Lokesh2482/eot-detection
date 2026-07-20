# End-of-Turn Detection

A causal, multilingual (English + Hindi) end-of-turn detector for voice agents. At every pause in user speech, it predicts the probability that the user's turn has ended — using only acoustic features computed from audio *before* the pause, with no future audio and no pretrained models.

**Start here:** [`starter/starter/SUMMARY.html`](starter/starter/SUMMARY.html) — full methodology, results, and compliance notes.

---

## Results

## Results

Mean response delay (ms) at ≤5% interrupted turns — lower is better.

| | English | Hindi |
|---|---:|---:|
| Final Model | 415 ms | 160 ms |
| Baseline | ~1600 ms | ~1600 ms |

Note: Latency on unseen test conversations may be higher than the values reported above, as the model is expected to generalize to previously unseen speakers, recording conditions, and conversational patterns.
---

## Repository Structure

## Repository Structure

```text
.
├── eot_data/
│   └── eot_data/
│       ├── english/
│       │   ├── audio/                  # English audio (.wav) files
│       │   ├── labels.csv              # Pause annotations
│       │   └── predictions.csv         # English predictions
│       └── hindi/
│           ├── audio/                  # Hindi audio (.wav) files
│           ├── labels.csv              # Pause annotations
│           └── predictions.csv         # Hindi predictions
│
├── starter/
│   └── starter/
│       ├── baseline.py                 # Silence-only baseline
│       ├── combine_predictions.py      # Utility to merge English & Hindi predictions
│       ├── ensemble.py                 # From-scratch NumPy randomized tree ensemble
│       ├── features.py                 # 56 causal acoustic feature extractor
│       ├── find_worst_errors.py        # Error analysis utility
│       ├── train.py                    # Grouped cross-validation & final training
│       ├── predict.py                  # Inference pipeline (no retraining)
│       ├── score.py                    # Official evaluation script
│       ├── model.pkl                   # Final multilingual model
│       ├── model_hindi_only.pkl        # Hindi-only experimental model
│       ├── predictions_english.csv     # English predictions
│       ├── predictions_hindi.csv       # Hindi predictions
│       ├── predictions.csv             # Combined multilingual predictions
│       ├── oof_*.csv                   # Out-of-fold predictions from validation experiments
│       ├── RUNLOG.md                   # Experiment log
│       ├── NOTES.md                    # Design choices, limitations & future work
│       ├── SUMMARY.html                # Project summary
│       ├── README.md                   # Project documentation
│       └── .gitignore
│
└── eot_data.zip                        # Dataset archive
```

---

## How it works

1. **Features** (`features.py`) — for each pause, extracts 56 features from the ~1.5s of audio strictly before `pause_start`: multi-scale energy trajectory, speaker-relative pitch (F0) dynamics, voicing continuity, and spectral shape (centroid, flux, flatness, zero-crossing rate).
2. **Model** (`ensemble.py`) — a randomized tree ensemble (ExtraTrees-style), implemented from scratch in NumPy, trained with class balancing.
3. **Training** (`train.py`) — turn-grouped 5-fold cross-validation over English + Hindi combined, evaluated with the official scorer per language. Final model refit on all data and saved to `model.pkl`.
4. **Inference** (`predict.py`) — loads `model.pkl` and scores a new folder with the same schema. No retraining.

---

## Reproducing

```bash
cd starter/starter

# Train (grouped CV + final model fit)
python train.py --data_dir ../../eot_data/eot_data/english ../../eot_data/eot_data/hindi

# Predict on a language folder
python predict.py --data_dir ../../eot_data/eot_data/english --out predictions_english.csv

# Score against the official metric
python score.py --data_dir ../../eot_data/eot_data/english --pred predictions_english.csv
```

---

## Constraints followed

- CPU-only, no GPU
- No pretrained models or downloaded weights
- No external datasets
- Strictly causal: features never use audio after `pause_start`
├── audio/ Hindi WAV files
├── labels.csv pause annotations
└── predictions.csv Hindi predictions (required deliverable)
