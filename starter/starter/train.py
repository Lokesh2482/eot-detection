"""Grouped OOF training + final model fit for the causal end-of-turn pipeline.

Trains on all supplied language folders together (multilingual, since the
hidden test set is mostly Hindi and features.py was built to be
speaker-relative/language-agnostic). Does two things:

  1. Runs turn-grouped 5-fold CV over the combined data and reports the
     official scorer result SEPARATELY for each language folder — this is
     the honest, leakage-free estimate of hidden-test performance.
  2. Refits the identical model on ALL available rows (both languages) and
     pickles it to `model.pkl`. predict.py loads this file at inference
     time; it never retrains.

Usage:
    python train.py --data_dir eot_data/english eot_data/hindi
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
from dataclasses import dataclass

import numpy as np

from features import extract_causal_features, feature_names, load_wav
from score import score

from ensemble import RandomizedTreeEnsemble


def _group_folds(groups, n_folds):
    """Create deterministic folds with whole turns kept together.

    Greedy assignment by group size balances the number of pause rows while
    preserving the hard no-shared-turn rule.
    """
    groups = np.asarray(groups)
    unique_groups, counts = np.unique(groups, return_counts=True)
    if n_folds < 2 or n_folds > len(unique_groups):
        raise ValueError(f"folds must be between 2 and {len(unique_groups)}")
    ordered = sorted(zip(unique_groups, counts), key=lambda item: (-item[1], item[0]))
    fold_groups = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    for group, count in ordered:
        fold = int(np.argmin(fold_sizes))
        fold_groups[fold].append(group)
        fold_sizes[fold] += count
    for held_out_groups in fold_groups:
        test_mask = np.isin(groups, held_out_groups)
        yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def _binary_auc(y, scores):
    """Tie-correct ROC AUC implemented without external ML dependencies."""
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[y == 1].sum() - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _load_labeled_data(data_dir):
    """Build feature rows for one labelled language folder."""
    labels_path = os.path.join(data_dir, "labels.csv")
    with open(labels_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"turn_id", "audio_file", "pause_index", "pause_start", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{labels_path} does not contain the required training columns")

    cache = {}
    features, labels, groups, keys = [], [], [], []
    for row in rows:
        if row["label"] not in {"hold", "eot"}:
            raise ValueError(f"unexpected label for {row['turn_id']}: {row['label']!r}")
        audio_path = os.path.join(data_dir, row["audio_file"])
        if audio_path not in cache:
            cache[audio_path] = load_wav(audio_path)
        x, sr = cache[audio_path]
        features.append(extract_causal_features(x, sr, float(row["pause_start"])))
        labels.append(1 if row["label"] == "eot" else 0)
        groups.append(row["turn_id"])
        keys.append((row["turn_id"], int(row["pause_index"])))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(groups),
        keys,
    )


def _load_multiple(data_dirs):
    """Load and concatenate every supplied language folder into one dataset.

    Also returns a `sources` array (same length as `keys`) recording which
    data_dir each row came from, so later we can score each language on its
    own even though the model itself is trained on everything combined.
    """
    all_x, all_y, all_groups, all_keys, all_sources = [], [], [], [], []
    for data_dir in data_dirs:
        x, y, groups, keys = _load_labeled_data(data_dir)
        all_x.append(x)
        all_y.append(y)
        all_groups.append(groups)
        all_keys.extend(keys)
        all_sources.extend([data_dir] * len(keys))
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    sources = np.asarray(all_sources)
    return x, y, groups, all_keys, sources


def _write_predictions(path, keys, probabilities):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for (turn_id, pause_index), probability in zip(keys, probabilities):
            writer.writerow([turn_id, pause_index, f"{float(probability):.6f}"])


def _append_runlog(path, data_dir, output_path, auc, accuracy, result, top_features, weak_features):
    """Record the exact OOF scorer run required by the assignment."""
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as handle:
        if new_file:
            handle.write("# Run log\n\n")
        handle.write(f"## Phase 2 - grouped OOF randomized-tree, multilingual fit ({data_dir})\n\n")
        handle.write(f"- Data: `{data_dir}`\n")
        handle.write(f"- Predictions: `{output_path}`\n")
        handle.write(f"- OOF accuracy: {accuracy:.3f}; tie-correct AUC: {auc:.3f}.\n")
        handle.write(
            "- Official score: "
            f"{result['latency'] * 1000:.0f} ms at {result['cutoff'] * 100:.1f}% "
            f"interrupted turns (threshold={result['threshold']}, "
            f"delay={result['delay'] * 1000:.0f} ms).\n"
        )
        handle.write(
            "- Change: trained one multilingual model on English+Hindi combined "
            "(turn-grouped OOF CV, class-balanced NumPy randomized tree ensemble), "
            "scored per-language; final model refit on all rows and pickled for predict.py.\n"
        )
        handle.write("- Strongest mean impurity features: " + ", ".join(top_features) + ".\n")
        handle.write("- Weakest mean impurity features: " + ", ".join(weak_features) + ".\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", nargs="+", required=True,
        help="one or more labelled folders, e.g. eot_data/english eot_data/hindi",
    )
    parser.add_argument("--out_prefix", default="oof", help="prefix for per-language OOF prediction CSVs")
    parser.add_argument("--runlog", default="RUNLOG.md")
    parser.add_argument("--model_out", default="model.pkl", help="where to pickle the final model for predict.py")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n_trees", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    x, y, groups, keys, sources = _load_multiple(args.data_dir)
    names = feature_names()
    if x.shape[1] != len(names):
        raise RuntimeError("feature matrix and feature schema disagree")

    # --- turn-grouped OOF CV over the COMBINED (multilingual) data ---
    oof = np.zeros(len(y), dtype=np.float32)
    fold_importances = []
    for fold_number, (train_indices, test_indices) in enumerate(_group_folds(groups, args.folds), start=1):
        model = RandomizedTreeEnsemble(
            n_trees=args.n_trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed + fold_number,
        ).fit(x[train_indices], y[train_indices])
        oof[test_indices] = model.predict_proba(x[test_indices])
        fold_importances.append(model.feature_importances_)
        fold_accuracy = float(np.mean((oof[test_indices] >= 0.5) == y[test_indices]))
        print(
            f"fold={fold_number} train_turns={len(np.unique(groups[train_indices]))} "
            f"test_turns={len(np.unique(groups[test_indices]))} accuracy={fold_accuracy:.3f}"
        )

    mean_importance = np.mean(np.asarray(fold_importances), axis=0)
    order = np.argsort(mean_importance)
    top_features = [f"{names[i]} ({mean_importance[i]:.4f})" for i in order[-10:][::-1]]
    weak_features = [f"{names[i]} ({mean_importance[i]:.4f})" for i in order[:10]]
    print("top feature importances:")
    for description in top_features:
        print(f"  {description}")
    print("weakest feature importances:")
    for description in weak_features:
        print(f"  {description}")

    # --- score each language SEPARATELY using its own OOF slice ---
    for data_dir in args.data_dir:
        mask = sources == data_dir
        sub_keys = [k for k, m in zip(keys, mask) if m]
        sub_oof = oof[mask]
        sub_y = y[mask]
        lang_tag = os.path.basename(os.path.normpath(data_dir))
        out_path = f"{args.out_prefix}_{lang_tag}.csv"
        _write_predictions(out_path, sub_keys, sub_oof)

        labels_csv = os.path.join(data_dir, "labels.csv")
        official = score(labels_csv, out_path)
        auc = _binary_auc(sub_y, sub_oof)
        accuracy = float(np.mean((sub_oof >= 0.5) == sub_y))
        print(f"[{lang_tag}] OOF rows={mask.sum()} turns={len(np.unique(groups[mask]))} "
              f"accuracy={accuracy:.3f} AUC={auc:.3f}")
        print(
            f"[{lang_tag}] OFFICIAL OOF SCORE: "
            f"{official['latency'] * 1000:.0f} ms, "
            f"cutoff={official['cutoff'] * 100:.1f}%, "
            f"threshold={official['threshold']}, delay={official['delay'] * 1000:.0f} ms"
        )
        _append_runlog(args.runlog, data_dir, out_path, auc, accuracy, official, top_features, weak_features)

    # --- final model: fit on EVERY row from every supplied folder, save it ---
    final_model = RandomizedTreeEnsemble(
        n_trees=args.n_trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
    ).fit(x, y)
    # predict_proba only needs .trees; drop the cached training arrays so the
    # pickle stays small and doesn't ship raw training data unnecessarily.
    final_model._x = None
    final_model._y = None
    final_model._sample_weight = None
    with open(args.model_out, "wb") as handle:
        pickle.dump(final_model, handle)
    print(f"saved final model trained on {len(y)} rows across {len(args.data_dir)} folder(s) -> {args.model_out}")


if __name__ == "__main__":
    main()