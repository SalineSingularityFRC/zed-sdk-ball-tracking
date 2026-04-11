"""Fit a ridge regression of (miss_r, miss_theta) against features derived
from the robot/turret state at the moment of each shot.

Standalone — does not import ZED, VTK, or NetworkTables. Run after
collecting shots.csv via the tracker."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error


FEATURE_NAMES = [
    "range",
    "inv_range",
    "sin_bearing",
    "cos_bearing",
    "v_radial",
    "v_tangential",
    "omega",
    "sin_turret_angle",
    "cos_turret_angle",
]


def featurize(row: dict) -> list[float]:
    rng = float(row["range_to_target"])
    bearing = float(row["bearing_to_target"])
    turret_angle = float(row["turret_angle"])
    return [
        rng,
        1.0 / rng if rng > 1e-6 else 0.0,
        math.sin(bearing),
        math.cos(bearing),
        float(row["v_radial"]),
        float(row["v_tangential"]),
        float(row["omega"]),
        math.sin(turret_angle),
        math.cos(turret_angle),
    ]


def load_shots(path: Path) -> list[dict]:
    with open(path, "r") as fp:
        return list(csv.DictReader(fp))


def fit(rows: list[dict], alpha: float) -> dict:
    X = np.array([featurize(r) for r in rows], dtype=float)
    y_r = np.array([float(r["miss_r"]) for r in rows], dtype=float)
    y_th = np.array([float(r["miss_theta"]) for r in rows], dtype=float)

    model_r = Ridge(alpha=alpha).fit(X, y_r)
    model_th = Ridge(alpha=alpha).fit(X, y_th)

    return {
        "n_samples": len(rows),
        "alpha": alpha,
        "feature_names": FEATURE_NAMES,
        "miss_r": {
            "intercept": float(model_r.intercept_),
            "coefs": [float(c) for c in model_r.coef_],
            "r2": float(r2_score(y_r, model_r.predict(X))),
            "mae": float(mean_absolute_error(y_r, model_r.predict(X))),
        },
        "miss_theta": {
            "intercept": float(model_th.intercept_),
            "coefs": [float(c) for c in model_th.coef_],
            "r2": float(r2_score(y_th, model_th.predict(X))),
            "mae": float(mean_absolute_error(y_th, model_th.predict(X))),
        },
    }


def print_report(result: dict) -> None:
    print(f"\nFit on {result['n_samples']} shots, alpha={result['alpha']}\n")
    for target in ("miss_r", "miss_theta"):
        block = result[target]
        print(f"  {target}: R2={block['r2']:.3f}  MAE={block['mae']:.4f}")
        print(f"    intercept = {block['intercept']:+.4f}")
        for name, c in zip(result["feature_names"], block["coefs"]):
            print(f"    {name:>16s} = {c:+.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default="shots.csv")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--save", type=str, default=None,
                        help="If set, write fitted coefficients to this JSON file")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"No shots file at {path}")
        return 1
    rows = load_shots(path)
    if len(rows) < 5:
        print(f"Only {len(rows)} shots in {path}; need at least 5 to fit")
        return 1

    result = fit(rows, alpha=args.alpha)
    print_report(result)

    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=2))
        print(f"Saved model to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
