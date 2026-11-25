"""Export the paired Phase 2C simulation as a single CSV."""

import csv
import json
from collections import Counter
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    notebook = json.loads((folder / "frontline_training_analytics.ipynb").read_text(encoding="utf-8"))
    source = next(cell["source"] for cell in notebook["cells"] if cell["id"] == "p2-simulate")
    scope = {}
    exec(compile("".join(source), "Phase 2C simulator", "exec"), scope)
    simulate = scope["simulate"]
    rows = []
    for baseline, signal in zip(simulate(probe_signal=0), simulate(probe_signal=24), strict=True):
        inputs = {key: value for key, value in baseline.items() if key != "delayed"}
        assert inputs == {key: value for key, value in signal.items() if key != "delayed"}
        rows.append({
            **inputs,
            "split": "development" if inputs["batch"] <= 14 else "held_out",
            "delayed_no_signal": baseline["delayed"],
            "delayed_with_signal": signal["delayed"],
        })
    assert len({row["worker"] for row in rows}) == len(rows) == 480
    assert Counter(row["batch"] for row in rows) == dict.fromkeys(range(1, 21), 24)
    assert Counter(row["split"] for row in rows) == {"development": 336, "held_out": 144}
    assert all(
        0 <= row["unrelated"] + row["interrupted"] + row["missing"] <= 2
        and all(0 <= row[field] <= 100 for field in
                ("pre", "post", "delayed_no_signal", "delayed_with_signal"))
        for row in rows
    )
    destination = folder / "dummy_training_data.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} synthetic records to {destination.name}")


if __name__ == "__main__":
    main()
