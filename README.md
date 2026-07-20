# End-of-Turn Detection

A causal, multilingual (English + Hindi) end-of-turn detector for voice agents. At every pause in user speech, it predicts the probability that the user's turn has ended — using only acoustic features computed from audio *before* the pause, with no future audio and no pretrained models.

**Start here:** [`starter/starter/SUMMARY.html`](starter/starter/SUMMARY.html) — full methodology, results, and compliance notes.

---

## Results

| Metric (mean response delay @ 5% interrupted turns) | English | Hindi |
|---|---|---|
| **This model (honest, out-of-fold)** | 1234 ms | 826 ms |
| Silence-only baseline | ~1600 ms | ~1600 ms |

Full experiment history, including a rejected Hindi-only training variant and reasoning for the final model choice, is in `RUNLOG.md`.

---

## Repo structure
starter/starter/ all code + required deliverables
├── features.py 56 causal acoustic features (energy, pitch, spectral)
├── ensemble.py from-scratch NumPy tree ensemble (shared by train/predict)
├── train.py grouped cross-validation + final model training
├── predict.py inference only, no retraining
├── score.py official scorer (provided)
├── baseline.py silence-only baseline (provided)
├── model.pkl final trained model
├── predictions_english.csv predictions on English data
├── predictions_hindi.csv predictions on Hindi data
├── predictions.csv combined predictions (both languages)
├── RUNLOG.md experiment log with all scores
├── NOTES.md model signal, failure modes, next steps
└── SUMMARY.html full write-up — start here

eot_data/eot_data/ provided audio + labels
├── english/
│ ├── audio/ English WAV files
│ ├── labels.csv pause annotations
│ └── predictions.csv English predictions (required deliverable)
└── hindi/


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
