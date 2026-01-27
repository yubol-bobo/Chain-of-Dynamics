#!/usr/bin/env python3
"""
Efficient CKD preprocessing to generate modeling-ready data.

This script mirrors the logic in src/data/preprocess.py but uses
column selection and vectorized operations for speed and lower memory use.
It produces a 1355-row cohort with 36 features across 8 time points.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


TIME_POINTS = 8


def build_time_cols(base_names, time_points=TIME_POINTS):
    return [f"{base}_Time_{t}" for base in base_names for t in range(time_points)]


def update_claim_columns_v2(df):
    new_columns = []
    for col in df.columns:
        if "_x.3" in col:
            new_columns.append(col.replace("_x.3", "_Time_6"))
        elif "_y.3" in col:
            new_columns.append(col.replace("_y.3", "_Time_7"))
        elif "_x.2" in col:
            new_columns.append(col.replace("_x.2", "_Time_4"))
        elif "_y.2" in col:
            new_columns.append(col.replace("_y.2", "_Time_5"))
        elif "_x.1" in col:
            new_columns.append(col.replace("_x.1", "_Time_2"))
        elif "_y.1" in col:
            new_columns.append(col.replace("_y.1", "_Time_3"))
        elif "_x" in col:
            new_columns.append(col.replace("_x", "_Time_0"))
        elif "_y" in col:
            new_columns.append(col.replace("_y", "_Time_1"))
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df


def main():
    parser = argparse.ArgumentParser(description="Fast CKD preprocessing")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/ckd_data/raw"),
        help="Raw CKD data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ckd_data/processed/ckd_merged_data_for_modeling.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/ckd_data/processed/ckd_merged_data_for_modeling.validation.json"),
        help="Validation report output path",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir

    # Base features (36 total across domains)
    base_cor = [
        "Diabetes",
        "Hyptsn",
        "CVD",
        "Anemia",
        "MA",
        "Prot",
        "Sec_Hyp",
        "Phos",
        "Atherosclerosis",
        "CHF",
        "Stroke",
        "CD",
        "MI",
        "FE",
        "MD",
        "ND",
        "S4_in_2years",
        "S5_in_2years",
    ]
    base_exp = [
        "n_claims_DR",
        "n_claims_I",
        "n_claims_O",
        "n_claims_P",
        "net_exp_DR",
        "net_exp_I",
        "net_exp_O",
        "net_exp_P",
    ]
    base_lab = [
        "Serum_Calcium",
        "eGFR",
        "Phosphorus",
        "Intact_PTH",
        "Hemoglobin",
        "UACR",
    ]
    base_demo = ["Age", "Gender", "Race", "BMI"]

    # ----------------------
    # 1) Load cohort list
    # ----------------------
    agg_cols = ["TMA_Acct"]
    agg_df = pd.read_csv(raw_dir / "Agg_data_24.csv", usecols=agg_cols)

    # ----------------------
    # 2) Load longitudinal comorbidities (exclude ESRD within obs window)
    # ----------------------
    long_com_cols = ["TMA_Acct"]
    # Note: longitudinal_comorbidity_df_2009_24.csv already uses normalized names
    # (Htn, Cvd, Athsc, SH) rather than Hyptsn/CVD/Atherosclerosis/Sec_Hyp.
    long_com_cols += build_time_cols(
        [
            "Transplant",
            "Dialysis",
            "Diabetes",
            "Htn",
            "Cvd",
            "Anemia",
            "MA",
            "Prot",
            "SH",
            "Phos",
            "Athsc",
            "CHF",
            "Stroke",
            "CD",
            "MI",
            "FE",
            "MD",
            "ND",
            "S4",
            "S5",
            "ESRD",
        ]
    )
    long_df = pd.read_csv(raw_dir / "longitudinal_comorbidity_df_2009_24.csv", usecols=long_com_cols)

    # Drop transplant/dialysis
    drop_cols = build_time_cols(["Transplant", "Dialysis"])
    long_df.drop(columns=drop_cols, inplace=True)

    # Remove patients with ESRD within observation period
    esrd_cols = build_time_cols(["ESRD"])
    long_df["ESRD_any"] = long_df[esrd_cols].max(axis=1)
    long_df = long_df[long_df["ESRD_any"] != 1].drop(columns=esrd_cols + ["ESRD_any"])

    # ----------------------
    # 3) Labs (drop Bicarbonate + Urine_Albumin)
    # ----------------------
    lab_cols = ["TMA_Acct"] + build_time_cols(base_lab)
    lab_df = pd.read_csv(raw_dir / "lab_test_longitudinal_df_2009_24.csv", usecols=lab_cols)
    lab_df = lab_df.replace(0, np.nan)

    # ----------------------
    # 4) Demographics
    # ----------------------
    demo_cols = ["TMA_Acct"] + build_time_cols(base_demo)
    demo_df = pd.read_csv(raw_dir / "agr_longitudinal_df_24.csv", usecols=demo_cols)

    # ----------------------
    # 5) Claims from final_df_24 (filter to cohort, reshape time)
    # ----------------------
    claim_suffixes = ["_x", "_y", "_x.1", "_y.1", "_x.2", "_y.2", "_x.3", "_y.3"]
    claim_cols = ["TMA_Acct", "ESRD"]
    for base in base_exp + base_cor:
        for suf in claim_suffixes:
            claim_cols.append(f"{base}{suf}")
    claim_df = pd.read_csv(raw_dir / "final_df_24.csv", usecols=claim_cols)

    # Filter claim cohort to Agg_data_24
    claim_df = claim_df[claim_df["TMA_Acct"].isin(agg_df["TMA_Acct"])]
    claim_df = update_claim_columns_v2(claim_df)

    # Normalize S4/S5 naming
    for s_old, s_new in [("S4_in_2years", "S4"), ("S5_in_2years", "S5")]:
        for t in range(TIME_POINTS):
            old = f"{s_old}_Time_{t}"
            new = f"{s_new}_Time_{t}"
            if old in claim_df.columns:
                claim_df.rename(columns={old: new}, inplace=True)

    # Split claims into expenditures and comorbidities
    exp_cols = ["TMA_Acct"] + build_time_cols(base_exp)
    cor_cols = ["TMA_Acct"] + build_time_cols(
        [
            "Diabetes",
            "Hyptsn",
            "CVD",
            "Anemia",
            "MA",
            "Prot",
            "Sec_Hyp",
            "Phos",
            "Atherosclerosis",
            "CHF",
            "Stroke",
            "CD",
            "MI",
            "FE",
            "MD",
            "ND",
            "S4",
            "S5",
        ]
    )
    claim_df_exp = claim_df[exp_cols].copy()
    claim_df_cor = claim_df[cor_cols].copy()
    claim_df_cor.columns = (
        claim_df_cor.columns.str.replace("Hyptsn", "Htn")
        .str.replace("CVD", "Cvd")
        .str.replace("Atherosclerosis", "Athsc")
        .str.replace("Sec_Hyp", "SH")
    )

    # ----------------------
    # 6) Merge longitudinal comorbidities with claim comorbidities
    # ----------------------
    merged = long_df.merge(claim_df_cor, on="TMA_Acct", suffixes=("_long", "_claim"))

    condition_cols = [c for c in claim_df_cor.columns if c != "TMA_Acct"]
    cor_data = {"TMA_Acct": merged["TMA_Acct"]}
    for col in condition_cols:
        a = merged[f"{col}_long"].fillna(0).astype(int)
        b = merged[f"{col}_claim"].fillna(0).astype(int)
        cor_data[col] = (a | b).astype(int)
    cor_df = pd.DataFrame(cor_data)

    # ----------------------
    # 7) Labels
    # ----------------------
    stage_df = pd.read_csv(raw_dir / "stage_merged_df_matched.csv", usecols=["TMA_Acct", "ESRD"])

    # ----------------------
    # 8) Merge everything
    # ----------------------
    data = (
        cor_df.merge(lab_df, on="TMA_Acct")
        .merge(demo_df, on="TMA_Acct")
        .merge(claim_df_exp, on="TMA_Acct")
        .merge(stage_df, on="TMA_Acct", how="left")
        .merge(claim_df[["TMA_Acct", "ESRD"]], on="TMA_Acct", how="left", suffixes=("_stage", "_claim"))
    )

    data["ESRD"] = ((data["ESRD_stage"] == 1) | (data["ESRD_claim"] == 1)).astype(int)
    data.drop(columns=["ESRD_stage", "ESRD_claim"], inplace=True)

    # ----------------------
    # 9) Sort columns by time block
    # ----------------------
    time_suffixes = sorted({col.split("_")[-1] for col in data.columns if "Time" in col})
    sorted_columns = ["TMA_Acct", "ESRD"]
    sorted_columns += [col for suffix in time_suffixes for col in data.columns if col.endswith(suffix)]
    data = data[sorted_columns]

    # ----------------------
    # 10) Sanity checks
    # ----------------------
    n_rows, n_cols = data.shape
    time_cols = [c for c in data.columns if "_Time_" in c]
    base_features = sorted({c[: c.rfind("_Time_")] for c in time_cols})
    assert n_rows == 1355, f"Unexpected cohort size: {n_rows}"
    assert len(base_features) == 36, f"Unexpected base feature count: {len(base_features)}"
    assert len(time_cols) == 36 * TIME_POINTS, f"Unexpected time feature count: {len(time_cols)}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"[SUCCESS] Wrote {args.output} ({n_rows} rows, {n_cols} cols)")

    # ----------------------
    # 11) Validation report
    # ----------------------
    time_points = TIME_POINTS
    time_cols = [c for c in data.columns if "_Time_" in c]
    base_features = sorted({c[: c.rfind("_Time_")] for c in time_cols})
    missing_counts = data[time_cols].isna().sum().to_dict()
    total_missing = int(data[time_cols].isna().sum().sum())

    report = {
        "rows": int(n_rows),
        "columns": int(n_cols),
        "time_points": int(time_points),
        "base_feature_count": int(len(base_features)),
        "base_features": base_features,
        "time_columns": int(len(time_cols)),
        "total_missing_time_values": total_missing,
        "missing_time_values_by_column": {
            k: int(v) for k, v in missing_counts.items() if v > 0
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(report).to_json(args.report, indent=2)
    print(f"[SUCCESS] Wrote validation report: {args.report}")


if __name__ == "__main__":
    main()
