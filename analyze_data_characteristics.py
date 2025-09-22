#!/usr/bin/env python3
"""
Comprehensive analysis of CKD and MIMIC dataset characteristics
to identify optimal preprocessing strategies for longitudinal data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_ckd_data():
    """Analyze CKD dataset characteristics."""
    print("="*80)
    print("                        CKD DATASET ANALYSIS")
    print("="*80)

    # Load CKD data
    ckd_path = "data/ckd_data/processed/ckd_merged_data_for_modeling.csv"
    if not Path(ckd_path).exists():
        print(f"[ERROR] CKD data not found at {ckd_path}")
        return None

    df = pd.read_csv(ckd_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB\n")

    # Basic statistics
    print("[BASIC STATISTICS]")
    print("-" * 40)
    print(f"Total patients: {len(df)}")
    print(f"Total features: {df.shape[1] - 2}")  # Excluding TMA_Acct and ESRD
    print(f"Target variable (ESRD): {df['ESRD'].value_counts().to_dict()}")
    print(f"Class imbalance ratio: {df['ESRD'].value_counts()[0] / df['ESRD'].value_counts()[1]:.2f}:1 (negative:positive)")

    # Time periods analysis
    time_periods = 8  # From 0 to 7
    features_per_period = (df.shape[1] - 2) // time_periods
    print(f"Time periods: {time_periods}")
    print(f"Features per time period: {features_per_period}")

    # Missing data analysis
    print(f"\n🔍 MISSING DATA ANALYSIS")
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

    # Feature types analysis
    print(f"\n📋 FEATURE TYPES ANALYSIS")
    print("-" * 40)

    # Extract unique feature names (without time suffix)
    unique_features = set()
    for col in feature_names:
        if '_Time_' in col:
            base_name = col.split('_Time_')[0]
            unique_features.add(base_name)

    print(f"Unique feature types: {len(unique_features)}")

    # Categorize features
    clinical_features = [f for f in unique_features if f in [
        'Diabetes', 'Htn', 'Cvd', 'Anemia', 'MA', 'Prot', 'SH', 'Phos',
        'Athsc', 'CHF', 'Stroke', 'CD', 'MI', 'FE', 'MD', 'ND', 'S4', 'S5'
    ]]

    lab_features = [f for f in unique_features if f in [
        'Serum_Calcium', 'eGFR', 'Phosphorus', 'Intact_PTH', 'Hemoglobin', 'UACR'
    ]]

    demographic_features = [f for f in unique_features if f in [
        'Age', 'Gender', 'Race', 'BMI'
    ]]

    utilization_features = [f for f in unique_features if f.startswith('n_claims') or f.startswith('net_exp')]

    print(f"  Clinical conditions: {len(clinical_features)}")
    print(f"  Laboratory values: {len(lab_features)}")
    print(f"  Demographics: {len(demographic_features)}")
    print(f"  Healthcare utilization: {len(utilization_features)}")

    # Data distribution analysis
    print(f"\n📈 DATA DISTRIBUTION ANALYSIS")
    print("-" * 40)

    # Analyze specific important features at Time_0
    key_features_t0 = ['eGFR_Time_0', 'Hemoglobin_Time_0', 'Age_Time_0', 'BMI_Time_0']
    available_features = [f for f in key_features_t0 if f in df.columns]

    for feature in available_features:
        if df[feature].notna().any():
            stats = df[feature].describe()
            print(f"  {feature}:")
            print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
            print(f"    Missing: {df[feature].isnull().sum()} ({df[feature].isnull().mean()*100:.1f}%)")

    return {
        'df': df,
        'missing_by_time': missing_by_time,
        'feature_categories': {
            'clinical': clinical_features,
            'lab': lab_features,
            'demographic': demographic_features,
            'utilization': utilization_features
        },
        'class_imbalance_ratio': df['ESRD'].value_counts()[0] / df['ESRD'].value_counts()[1],
        'total_missing_pct': missing_percentage
    }

def analyze_mimic_data():
    """Analyze MIMIC dataset characteristics."""
    print("\n" + "="*80)
    print("                       MIMIC-IV DATASET ANALYSIS")
    print("="*80)

    # Look for MIMIC processed data
    mimic_paths = [
        "data/mimic-iv-3.1/processed/mimic_merged_data_for_modeling.csv",
        "data/mimic-iv-3.1/processed/cohort_icu_mortality.csv",
        "data/mimic-iv-3.1/processed/processed_mimic_data.csv"
    ]

    mimic_path = None
    for path in mimic_paths:
        if Path(path).exists():
            mimic_path = path
            break

    if not mimic_path:
        print(f"[WARNING] MIMIC data not found at any of these paths:")
        for path in mimic_paths:
            print(f"  - {path}")
        return None

    print(f"Loading MIMIC data from: {mimic_path}")
    df = pd.read_csv(mimic_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB\n")

    # Basic statistics
    print("[BASIC STATISTICS]")
    print("-" * 40)
    print(f"Total patients: {len(df)}")
    print(f"Total features: {df.shape[1] - 1}")  # Excluding target

    # Identify target column
    target_cols = [col for col in df.columns if 'mort' in col.lower() or 'death' in col.lower() or 'outcome' in col.lower()]
    if target_cols:
        target_col = target_cols[0]
        print(f"Target variable ({target_col}): {df[target_col].value_counts().to_dict()}")
        if len(df[target_col].unique()) == 2:
            class_counts = df[target_col].value_counts()
            ratio = class_counts.iloc[0] / class_counts.iloc[1]
            print(f"Class imbalance ratio: {ratio:.2f}:1")
    else:
        print("Target variable: Not clearly identified")
        target_col = None

    # Missing data analysis
    print(f"\n🔍 MISSING DATA ANALYSIS")
    print("-" * 40)
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percentage = (total_missing / total_cells) * 100
    print(f"Total missing values: {total_missing:,} ({missing_percentage:.2f}%)")

    # Features with high missing rates
    missing_by_feature = df.isnull().mean() * 100
    high_missing = missing_by_feature[missing_by_feature > 50]
    if len(high_missing) > 0:
        print(f"Features with >50% missing data: {len(high_missing)}")
        for feature, pct in high_missing.head(10).items():
            print(f"  {feature}: {pct:.1f}%")

    return {
        'df': df,
        'target_col': target_col,
        'total_missing_pct': missing_percentage,
        'class_imbalance_ratio': ratio if target_col else None
    }

def recommend_preprocessing_strategies(ckd_analysis, mimic_analysis):
    """Provide recommendations for preprocessing strategies."""
    print("\n" + "="*80)
    print("                    PREPROCESSING RECOMMENDATIONS")
    print("="*80)

    print("🎯 KEY FINDINGS")
    print("-" * 40)

    if ckd_analysis:
        print(f"CKD Dataset:")
        print(f"  • Class imbalance: {ckd_analysis['class_imbalance_ratio']:.1f}:1 ratio")
        print(f"  • Missing data: {ckd_analysis['total_missing_pct']:.1f}% overall")
        print(f"  • Longitudinal structure: 8 time periods")

        # Analyze missing pattern progression
        missing_trends = ckd_analysis['missing_by_time']
        early_missing = np.mean([missing_trends[f'Time_{i}'] for i in range(3)])
        late_missing = np.mean([missing_trends[f'Time_{i}'] for i in range(5, 8)])
        print(f"  • Missing data increases over time: {early_missing:.1f}% → {late_missing:.1f}%")

    if mimic_analysis:
        print(f"\nMIMIC Dataset:")
        print(f"  • Class imbalance: {mimic_analysis['class_imbalance_ratio']:.1f}:1 ratio")
        print(f"  • Missing data: {mimic_analysis['total_missing_pct']:.1f}% overall")

    print(f"\n💡 RECOMMENDED STRATEGIES")
    print("-" * 40)

    print("1. MISSING DATA IMPUTATION:")
    print("   Current approach: Simple fillna(0) - PROBLEMATIC")
    print("   Recommended approaches:")
    print("   ✅ Forward Fill + Backward Fill for temporal continuity")
    print("   ✅ KNN imputation with temporal neighbors")
    print("   ✅ Multiple imputation (MICE) with time-aware features")
    print("   ✅ Linear interpolation for lab values")
    print("   ✅ Carry-forward for clinical conditions")

    print("\n2. CLASS IMBALANCE HANDLING:")
    print("   Current approach: TSMOTE - Good start but can be improved")
    print("   Recommended approaches:")
    print("   ✅ ADASYN (Adaptive Synthetic Sampling)")
    print("   ✅ SMOTENC for mixed data types")
    print("   ✅ Temporal-aware resampling")
    print("   ✅ Cost-sensitive learning (adjust class weights)")
    print("   ✅ Ensemble methods with balanced sampling")

    print("\n3. TEMPORAL DATA PREPROCESSING:")
    print("   ✅ Preserve temporal ordering in train/val/test splits")
    print("   ✅ Feature engineering: time-since-last-visit, trends")
    print("   ✅ Handle irregular time intervals")
    print("   ✅ Create rolling window features")

    print("\n4. FEATURE ENGINEERING:")
    print("   ✅ Derive trend features (increasing/decreasing patterns)")
    print("   ✅ Time-to-event features")
    print("   ✅ Interaction features between time periods")
    print("   ✅ Missing data indicators as features")

    print("\n5. NORMALIZATION STRATEGY:")
    print("   Current: StandardScaler per sample - May lose temporal relationships")
    print("   Recommended:")
    print("   ✅ Robust scaling (less sensitive to outliers)")
    print("   ✅ Per-feature normalization across all time points")
    print("   ✅ Min-Max scaling for bounded features")

    print(f"\n⚠️  CURRENT ISSUES LIKELY CAUSING POOR PERFORMANCE:")
    print("-" * 50)
    print("1. Zero-filling missing values destroys clinical meaning")
    print("2. Simple TSMOTE may not preserve temporal patterns")
    print("3. Extreme class imbalance not adequately addressed")
    print("4. No temporal feature engineering")
    print("5. Normalization approach may lose temporal relationships")

def main():
    """Main analysis function."""
    print("LONGITUDINAL HEALTHCARE DATA ANALYSIS")
    print("Analyzing CKD and MIMIC datasets for optimal preprocessing strategies\n")

    # Analyze CKD data
    ckd_analysis = analyze_ckd_data()

    # Analyze MIMIC data
    mimic_analysis = analyze_mimic_data()

    # Provide recommendations
    recommend_preprocessing_strategies(ckd_analysis, mimic_analysis)

    print(f"\n✅ Analysis completed!")
    print("📝 Next steps: Implement improved preprocessing pipeline based on recommendations")

if __name__ == "__main__":
    main()