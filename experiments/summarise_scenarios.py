"""Tabulate the scenario-suite results (docked / runs per scenario, disturbance, variant and round).

    python experiments/summarise_scenarios.py
"""

import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "scenarios"
ROUNDS = {"R1": "*.jsonl", "R2": "round2/*.jsonl", "R3": "round3/*.jsonl"}


def load(pattern):
    return [json.loads(line) for f in glob.glob(str(ROOT / pattern)) for line in open(f)]


def main():
    data = {k: load(v) for k, v in ROUNDS.items()}
    keys = sorted({(r["scenario"], r["disturbance"]) for rows in data.values() for r in rows})
    print("scenario       disturbance  variant  " + "  ".join(ROUNDS))
    for sc, di in keys:
        for v in ("C", "B"):
            cells = []
            for k in ROUNDS:
                rs = [r for r in data[k] if r["scenario"] == sc and r["disturbance"] == di and r["variant"] == v]
                cells.append(f"{sum(r['outcome'] == 'docked' for r in rs)}/{len(rs)}")
            print(f"{sc:14s} {di:12s} {v:7s}  " + "  ".join(f"{c:4s}" for c in cells))
    print()
    for k, rows in data.items():
        for v in ("C", "B"):
            nom = [r for r in rows if r["disturbance"] == "none" and r["variant"] == v]
            dis = [r for r in rows if r["disturbance"] != "none" and r["variant"] == v]
            print(f"{k} {v}: layouts {sum(r['outcome'] == 'docked' for r in nom)}/{len(nom)}, "
                  f"disturbances {sum(r['outcome'] == 'docked' for r in dis)}/{len(dis)}, "
                  f"collisions {sum(r['outcome'] == 'collision' for r in nom + dis)}")


if __name__ == "__main__":
    main()
