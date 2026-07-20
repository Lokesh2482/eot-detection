"""Inference entry point for the end-of-turn detector.

Loads the model already trained and saved by train.py (model.pkl) and scores
an unseen data folder with the same schema as eot_data/english or
eot_data/hindi. Does not retrain, does not look at pause_end or label.

Usage:
    python predict.py --data_dir eot_data/hindi --out predictions.csv
"""
import argparse
import csv
import os
import pickle

import numpy as np

from features import extract_causal_features, load_wav
# RandomizedTreeEnsemble must be importable under this exact name for
# pickle.load to reconstruct the model saved by train.py.
from ensemble import RandomizedTreeEnsemble  # noqa: F401


def _load_rows(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    with open(labels_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="folder with audio/ and labels.csv")
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument("--model", default="model.pkl", help="pickled model produced by train.py")
    args = parser.parse_args()

    with open(args.model, "rb") as handle:
        model = pickle.load(handle)

    rows = _load_rows(args.data_dir)

    cache = {}
    features, keys = [], []
    for row in rows:
        audio_path = os.path.join(args.data_dir, row["audio_file"])
        if audio_path not in cache:
            cache[audio_path] = load_wav(audio_path)
        x, sr = cache[audio_path]
        # Only pause_start is used, matching the causality rule — pause_end
        # and label (if present in this folder's labels.csv) are ignored.
        features.append(extract_causal_features(x, sr, float(row["pause_start"])))
        keys.append((row["turn_id"], int(row["pause_index"])))

    x = np.asarray(features, dtype=np.float32)
    probabilities = model.predict_proba(x)

    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for (turn_id, pause_index), probability in zip(keys, probabilities):
            writer.writerow([turn_id, pause_index, f"{float(probability):.6f}"])

    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()