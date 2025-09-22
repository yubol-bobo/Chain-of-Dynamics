#!/usr/bin/env python3
"""
Comprehensive analysis of CKD and MIMIC dataset characteristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_ckd_data():
    """Analyze CKD dataset characteristics."""
    print("="*80)
    print("                        CKD DATASET ANALYSIS")
    print("="*80)

    ckd_path = "data/ckd_data/processed/ckd_merged_data_for_modeling.csv"
    if not Path(ckd_path).exists():
        print(f"[ERROR] CKD data not found at {ckd_path}")
        return None

    df = pd.read_csv(ckd_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB\n")

    # Basic statistics
    print("BASIC STATISTICS")
    print("-" * 40)
    print(f"Total patients: {len(df)}")
    print(f"Total features: {df.shape[1] - 2}")
    target_counts = df['ESRD'].value_counts()
    print(f"Target variable (ESRD): {target_counts.to_dict()}")
    class_ratio = target_counts[0] / target_counts[1] if len(target_counts) == 2 else "N/A"
    print(f"Class imbalance ratio: {class_ratio:.2f}:1 (negative:positive)")

    # Time periods analysis
    time_periods = 8
    features_per_period = (df.shape[1] - 2) // time_periods
    print(f"Time periods: {time_periods}")
    print(f"Features per time period: {features_per_period}")

    # Missing data analysis
    print(f"\nMISSING DATA ANALYSIS")
    print("-" * 40)
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percentage = (total_missing / total_cells) * 100
    print(f"Total missing values: {total_missing:,} ({missing_percentage:.2f}%)")

    # Missing data patterns by time period
    missing_by_time = {}
    feature_names = [col for col in df.columns if col not in ['TMA_Acct', 'ESRD']]

    for t in range(time_periods):
        time_features = [col for col in feature_names if f'_Time_{t}' in col]
        if time_features:
            missing_count = df[time_features].isnull().sum().sum()
            total_values = len(df) * len(time_features)
            missing_pct = (missing_count / total_values) * 100
            missing_by_time[f'Time_{t}'] = missing_pct
            print(f"  Time {t}: {missing_pct:.2f}% missing")

    # Feature analysis
    print(f"\nFEATURE ANALYSIS")
    print("-" * 40)

    # Sample some key features
    key_features = ['eGFR_Time_0', 'Hemoglobin_Time_0', 'Age_Time_0']
    for feature in key_features:
        if feature in df.columns and df[feature].notna().any():
            stats = df[feature].describe()
            missing_pct = df[feature].isnull().mean() * 100
            print(f"{feature}:")
            print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
            print(f"  Missing: {missing_pct:.1f}%")

    return {
        'df': df,
        'missing_by_time': missing_by_time,
        'class_imbalance_ratio': class_ratio,
        'total_missing_pct': missing_percentage
    }

def analyze_mimic_data():
    """Analyze MIMIC dataset characteristics."""
    print("\n" + "="*80)
    print("                       MIMIC-IV DATASET ANALYSIS")
    print("="*80)

    mimic_paths = [
        "data/mimic-iv-3.1/processed/mimic_merged_data_for_modeling.csv",
        "data/mimic-iv-3.1/processed/cohort_icu_mortality.csv"
    ]

    mimic_path = None
    for path in mimic_paths:
        if Path(path).exists():
            mimic_path = path
            break

    if not mimic_path:
        print("[WARNING] MIMIC data not found")
        return None

    df = pd.read_csv(mimic_path)
    print(f"Dataset shape: {df.shape}")

    # Basic statistics
    print("\nBASIC STATISTICS")
    print("-" * 40)
    print(f"Total patients: {len(df)}")

    # Find target column
    target_cols = [col for col in df.columns if 'mort' in col.lower() or 'death' in col.lower()]
    if target_cols:
        target_col = target_cols[0]
        target_counts = df[target_col].value_counts()
        print(f"Target variable ({target_col}): {target_counts.to_dict()}")
        class_ratio = target_counts.iloc[0] / target_counts.iloc[1]
        print(f"Class imbalance ratio: {class_ratio:.2f}:1")
    else:
        target_col = None
        class_ratio = None

    # Missing data
    total_missing = df.isnull().sum().sum()
    missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
    print(f"Total missing: {missing_percentage:.2f}%")

    return {
        'df': df,
        'target_col': target_col,
        'class_imbalance_ratio': class_ratio,
        'total_missing_pct': missing_percentage
    }

def provide_recommendations(ckd_analysis, mimic_analysis):
    """Provide preprocessing recommendations."""
    print("\n" + "="*80)
    print("                    PREPROCESSING RECOMMENDATIONS")
    print("="*80)

    print("KEY FINDINGS:")
    print("-" * 40)

    if ckd_analysis:
        print(f"CKD Dataset:")
        print(f"  - Class imbalance: {ckd_analysis['class_imbalance_ratio']:.1f}:1 ratio")
        print(f"  - Missing data: {ckd_analysis['total_missing_pct']:.1f}% overall")

        # Check missing data progression
        missing_times = ckd_analysis['missing_by_time']
        early_avg = np.mean([missing_times[f'Time_{i}'] for i in range(3)])
        late_avg = np.mean([missing_times[f'Time_{i}'] for i in range(5, 8)])
        print(f"  - Missing increases over time: {early_avg:.1f}% -> {late_avg:.1f}%")

    if mimic_analysis:
        print(f"MIMIC Dataset:")
        print(f"  - Class imbalance: {mimic_analysis['class_imbalance_ratio']:.1f}:1 ratio")
        print(f"  - Missing data: {mimic_analysis['total_missing_pct']:.1f}% overall")

    print(f"\nRECOMMENDED IMPROVEMENTS:")
    print("-" * 40)

    print("1. MISSING DATA IMPUTATION:")
    print("   CURRENT: Simple fillna(0) - PROBLEMATIC")
    print("   RECOMMENDED:")
    print("   - Forward/backward fill for temporal continuity")
    print("   - KNN imputation with temporal neighbors")
    print("   - Linear interpolation for lab values")
    print("   - Carry-forward for clinical conditions")

    print("\n2. CLASS IMBALANCE:")
    print("   CURRENT: TSMOTE")
    print("   RECOMMENDED:")
    print("   - ADASYN (Adaptive Synthetic Sampling)")
    print("   - Temporal-aware resampling")
    print("   - Cost-sensitive learning")
    print("   - Ensemble methods with balanced sampling")

    print("\n3. FEATURE ENGINEERING:")
    print("   - Time-since-last-visit features")
    print("   - Trend/slope features")
    print("   - Rolling window statistics")
    print("   - Missing data indicator features")

    print("\n4. NORMALIZATION:")
    print("   CURRENT: StandardScaler per sample")
    print("   RECOMMENDED: Robust scaling per feature across time")

    print(f"\nCURRENT ISSUES CAUSING POOR PERFORMANCE:")
    print("-" * 50)
    print("1. Zero-filling destroys clinical meaning")
    print("2. Extreme class imbalance not handled optimally")
    print("3. Missing temporal feature engineering")
    print("4. Sub-optimal normalization strategy")

def main():
    print("LONGITUDINAL HEALTHCARE DATA ANALYSIS")
    print("="*50)

    ckd_analysis = analyze_ckd_data()
    mimic_analysis = analyze_mimic_data()
    provide_recommendations(ckd_analysis, mimic_analysis)

    print(f"\nANALYSIS COMPLETED!")

if __name__ == "__main__":
    main()