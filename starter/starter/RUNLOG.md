# Run Log

## Phase 0 – Setup & Baseline
- Ran `baseline.py` (silence-only, p_eot=1 always). Reference: ~1600ms mean delay @ 5% cutoff - this is the number to beat.

## Phase 1 – Feature Engineering
- Implemented `features.py`: 56 strictly causal features (multi-scale energy, speaker-relative F0, spectral centroid/flatness/flux, ZCR, voicing/activity timing). Verified causality (only audio before `pause_start` is used) and no label leakage.
- Change: froze `features.py` after this point; all further iteration was on the training pipeline, not the feature set.

## Phase 2 – Training Pipeline
- Replaced starter `train.py`'s single random split + LogisticRegression with turn-grouped 5-fold CV, evaluated by ROC-AUC.
- Implemented `RandomizedTreeEnsemble` (NumPy, from scratch — no sklearn dependency in the runtime environment). Moved shared model classes into `ensemble.py` so `train.py` and `predict.py` can both import them and `pickle` resolves correctly.
- Change: trained on English + Hindi **combined** (multilingual), reasoning that hidden test set is "mostly Hindi" and shared prosodic signal should generalize. Verified this against a Hindi-only run (see below).

## Phase 3 – Evaluation

**Out-of-fold (OOF) — honest, leakage-free estimate of hidden-test performance:**
- Combined model (English+Hindi), scored per language:
  - English: **1234 ms** @ 5.0% interrupted turns (threshold=0.5, delay=750ms), AUC=0.610
  - Hindi: **826 ms** @ 5.0% interrupted turns (threshold=0.35, delay=750ms), AUC=0.659
- Hindi-only model (trained only on Hindi, 248 rows):
  - Hindi: **850 ms** @ 5.0% interrupted turns (threshold=0.05, delay=850ms), AUC=0.718
- **Decision: kept the combined model.** It scores better on Hindi (826ms vs 850ms) despite the Hindi-only model having higher OOF AUC — likely because the combined model has 2x the training data and the scorer optimizes ranking at a specific operating point, not raw AUC. Combined model also benefits from more turns for the tree ensemble's bootstrap sampling.

**In-sample (`predict.py` on `model.pkl`, which was fit on all 496 rows including these) — pipeline sanity check only, NOT predictive of hidden-test performance since the model has already seen this data:**
- English: 415 ms @ 2.0% interrupted turns (threshold=0.55, delay=100ms), AUC=0.981
- Hindi: 160 ms @ 3.0% interrupted turns (threshold=0.5, delay=100ms), AUC=0.995
- These confirm the train → save → load → infer pipeline works correctly end-to-end. They are expected to look much better than OOF because the model memorized these exact rows during the final fit.

## Phase 4 – Final Submission
- `predict.py`: loads `model.pkl`, extracts features, runs inference. No retraining. Interface: `python predict.py --data_dir <folder> --out predictions.csv`.
- Final predictions generated: `predictions_english.csv`, `predictions_hindi.csv` (from the combined `model.pkl`).
- Feature importance (combined model, top): `energy_last_active_run_s`, `spectral_flux_0500ms_mean`, `f0_1500ms_slope_semitones_s`, `f0_0500ms_slope_semitones_s`, `energy_0500ms_mean_db`. Weakest: `f0_reference_available`, `context_s`, `context_is_short` (near-zero importance — candidates for pruning with more time, not removed here since it didn't hurt or help materially).

