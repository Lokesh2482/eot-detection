"""find_worst_errors.py — pulls highest-confidence wrong OOF predictions.

Usage:
    python find_worst_errors.py <labels_csv> <oof_csv> [n]
"""
import csv
import sys


def worst_errors(labels_csv, oof_csv, n=10):
    preds = {}
    with open(oof_csv) as f:
        for r in csv.DictReader(f):
            preds[(r["turn_id"], int(r["pause_index"]))] = float(r["p_eot"])

    rows = []
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            key = (r["turn_id"], int(r["pause_index"]))
            if key not in preds:
                continue
            p = preds[key]
            true_eot = 1 if r["label"] == "eot" else 0
            error = abs(p - true_eot)
            rows.append((error, r["turn_id"], r["pause_index"], r["label"], p,
                         r["pause_start"], r["pause_end"]))

    rows.sort(reverse=True)
    print(f"{'turn_id':20} {'pause':6} {'true':6} {'p_eot':8} {'start':8} {'end':8}")
    for error, turn_id, idx, label, p, start, end in rows[:n]:
        print(f"{turn_id:20} {idx:<6} {label:6} {p:<8.3f} {start:<8} {end:<8}")


if __name__ == "__main__":
    labels_csv, oof_csv = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    worst_errors(labels_csv, oof_csv, n)