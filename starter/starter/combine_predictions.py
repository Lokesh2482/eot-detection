"""combine_predictions.py — merges per-language predictions into one file."""
import csv

files = ["predictions_english.csv", "predictions_hindi.csv"]
out_path = "predictions.csv"

rows = []
for path in files:
    with open(path, newline="", encoding="utf-8") as f:
        rows.extend(list(csv.DictReader(f)))

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows)} predictions -> {out_path}")