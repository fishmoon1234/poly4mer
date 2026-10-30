# Proposed replacement for analyze_fire_properties.py — no files were changed.
#
# Run:
# python analyze_fire_properties.py results/sweep_ridge/a*_k*_lam0.001.csv
#
# Ranking: for each property, select the ridge configuration with the lowest
# mean absolute relative error for each polymer, then report the best 20 polymers.
# Repeated experimental conditions are averaged within each polymer/configuration.
#
# These are full-data training predictions, not held-out predictions.
# predict_fire_properties.py fits on all available experimental rows per property.
#
# IMPORTANT: that script does not save prediction heads. Actual head paths cannot
# be recovered from these CSVs. They are explicitly reported as NOT_SAVED.
# SAE paths below are verified against the corresponding training log and disk.

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SMILES = "smiles_canonicalized"
NAME = re.compile(r"a(\d+)_k(\d+)_lam([0-9.eE+-]+)\.csv")


def main():
    paths = sorted({
        Path(match).resolve()
        for pattern in sys.argv[1:]
        for match in glob.glob(pattern)
    })

    if not paths:
        raise SystemExit(
            "Usage: python analyze_fire_properties.py "
            "'results/ridge_final/a*_k*_lam0.001.csv'"
        )

    frames = []

    for path in paths:
        match = NAME.fullmatch(path.name)
        if not match or path.parent.name != "ridge_final":
            raise ValueError(f"Expected a CSV from results/ridge_final: {path}")

        alpha, k, lam_text = match.groups()
        log_path = ROOT / "logs" / "ridge_final" / f"{path.stem}.log"

        if not log_path.is_file():
            raise FileNotFoundError(f"Cannot verify training run: {log_path}")

        log = log_path.read_text()
        summary = re.search(
            rf"alpha={alpha}\s+k={k}\s+solver=ridge\b", log
        )
        if not summary or "wrote per-row predictions to" not in log:
            raise ValueError(f"Cannot verify completed full-data ridge run: {log_path}")

        # This identifies the full-data predictor's log format.
        # Cross-validation logs must not be substituted.
        if "=== tig ===" not in log or re.search(r"\bfold\b", log, re.I):
            raise ValueError(f"Unexpected training log format: {log_path}")

        sae_match = re.search(r"Load SAE from\s+(.+)", log)
        if not sae_match:
            raise ValueError(f"SAE path missing from log: {log_path}")

        sae_path = Path(sae_match.group(1).strip())
        if not sae_path.is_absolute():
            sae_path = ROOT / sae_path
        sae_path = sae_path.resolve()

        if not sae_path.is_file():
            raise FileNotFoundError(sae_path)

        df = pd.read_csv(path)
        required = {SMILES, "property", "predicted"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path}: missing {required - set(df.columns)}")

        head_matches = re.findall(r"Saved prediction head:\s*(.+)", log)
        
        for prop, group in df.groupby("property", sort=True):
            expected_name = f"{prop}_alpha{alpha}_k{k}_ridge.pt"
            head_paths = []

            for saved_path in head_matches:
                candidate = Path(saved_path.strip())
                if not candidate.is_absolute():
                    candidate = ROOT / candidate
                if candidate.name == expected_name:
                    head_paths.append(candidate.resolve())

            if len(head_paths) != 1:
                raise ValueError(
                    f"Expected one saved head for {prop} in {log_path}; "
                    f"found {len(head_paths)}"
                )

            head_path = head_paths[0]
            if not head_path.is_file():
                raise FileNotFoundError(f"Prediction head missing: {head_path}")
                    
            if prop not in df.columns:
                raise ValueError(f"{path}: ground-truth column {prop!r} missing")

            group = group.copy()
            group["ground_truth"] = pd.to_numeric(group[prop], errors="raise")
            group["predicted"] = pd.to_numeric(group["predicted"], errors="raise")

            valid = np.isfinite(group[["ground_truth", "predicted"]]).all(axis=1)
            if not valid.all():
                raise ValueError(f"{path}: nonfinite values for {prop}")

            if group[SMILES].isna().any():
                raise ValueError(f"{path}: missing polymer SMILES for {prop}")

            # Relative error is undefined when ground truth equals zero.
            if group["ground_truth"].eq(0).any():
                raise ValueError(f"{path}: zero ground truth for {prop}")

            group["relative_error_pct"] = (
                (group["predicted"] - group["ground_truth"]).abs()
                / group["ground_truth"].abs()
                * 100
            )

            table = group.groupby(SMILES, as_index=False).agg(
                ground_truth=("ground_truth", "mean"),
                predicted=("predicted", "mean"),
                relative_error_pct=("relative_error_pct", "mean"),
                experimental_rows=("ground_truth", "size"),
            )

            table["property"] = prop
            table["alpha"] = int(alpha)
            table["k"] = int(k)
            table["ridge_lambda"] = float(lam_text)
            table["sae_path"] = str(sae_path)
            table["head_path"] = str(head_path)
            table["prediction_csv"] = str(path)
            frames.append(table)

    combined = pd.concat(frames, ignore_index=True)

    for prop, group in combined.groupby("property", sort=True):
        # Keep one real configuration per polymer. Never average predictions
        # across configurations while reporting a single model's path.
        top20 = (
            group.sort_values(
                ["relative_error_pct", "alpha", "k", "ridge_lambda", SMILES],
                kind="stable",
            )
            .drop_duplicates(SMILES, keep="first")
            .head(20)
            .reset_index(drop=True)
        )
        top20.insert(0, "rank", np.arange(1, len(top20) + 1))

        print(f"\n{'=' * 100}")
        print(f"{prop.upper()}: TOP {len(top20)} POLYMERS — LOWEST TRAINING ERROR")
        print("Values are means across experimental rows within each polymer.")
        print("Error is mean row-wise absolute relative error (%).")
        print(top20.to_string(index=False, float_format=lambda value: f"{value:.8g}"))

        output = Path(f"table_a2_top20_ridge_{prop}.csv")
        top20.to_csv(output, index=False)
        print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()