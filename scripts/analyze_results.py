#!/usr/bin/env python3
"""
Results Analysis Script for Chain-of-Dynamics Project

This script collects test metrics from all trained models and generates
comprehensive summary tables for both CKD and MIMIC-IV datasets.

Usage:
    python scripts/analyze_results.py

Output:
    - results/summary_ckd_results.csv
    - results/summary_mimic_results.csv
    - results/summary_ckd_results.xlsx
    - results/summary_mimic_results.xlsx
    - Console output with formatted tables
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_model_metrics(results_dir, model_name):
    """
    Load test metrics for a specific model from its results directory.

    Args:
        results_dir (str): Path to results directory (e.g., './results/mimic/')
        model_name (str): Name of the model (e.g., 'bilstm', 'retain', etc.)

    Returns:
        dict: Dictionary containing model metrics or None if not found
    """
    metrics_file = os.path.join(results_dir, f"{model_name}_test_metrics.yaml")

    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found for {model_name} in {results_dir}")
        return None

    try:
        with open(metrics_file, 'r') as f:
            metrics = yaml.safe_load(f)

        # Extract the main metrics (assuming standard format)
        if isinstance(metrics, dict):
            return {
                'F1': metrics.get('f1', metrics.get('F1', 'N/A')),
                'AUROC': metrics.get('auroc', metrics.get('AUROC', metrics.get('auc', 'N/A'))),
                'Accuracy': metrics.get('accuracy', metrics.get('Accuracy', 'N/A')),
                'Precision': metrics.get('precision', metrics.get('Precision', 'N/A')),
                'Recall': metrics.get('recall', metrics.get('Recall', 'N/A')),
                'Specificity': metrics.get('specificity', metrics.get('Specificity', 'N/A'))
            }
        else:
            print(f"Warning: Unexpected metrics format for {model_name}")
            return None

    except Exception as e:
        print(f"Error loading metrics for {model_name}: {str(e)}")
        return None


def load_hyperparameter_summary(results_dir, model_name):
    """
    Load hyperparameter search summary for additional context.

    Args:
        results_dir (str): Path to results directory
        model_name (str): Name of the model

    Returns:
        dict: Best hyperparameters or None if not found
    """
    summary_file = os.path.join(results_dir, f"{model_name}_hypersearch_summary.yaml")

    if not os.path.exists(summary_file):
        return None

    try:
        with open(summary_file, 'r') as f:
            summary = yaml.safe_load(f)
        return summary.get('best_params', {})
    except Exception as e:
        print(f"Error loading hyperparameter summary for {model_name}: {str(e)}")
        return None


def collect_all_results(base_results_dir="./results"):
    """
    Collect results from all models for both datasets.

    Args:
        base_results_dir (str): Base directory containing results

    Returns:
        tuple: (ckd_results, mimic_results) as DataFrames
    """
    # Define all models and datasets
    models = ['bilstm', 'retain', 'transformer', 'adacare', 'stagenet', 'coi']
    datasets = {
        'ckd': os.path.join(base_results_dir, 'ckd'),
        'mimic': os.path.join(base_results_dir, 'mimic')
    }

    results = {'ckd': [], 'mimic': []}

    for dataset_name, dataset_dir in datasets.items():
        print(f"\n=== Collecting results for {dataset_name.upper()} dataset ===")

        if not os.path.exists(dataset_dir):
            print(f"Warning: Results directory not found: {dataset_dir}")
            continue

        for model in models:
            print(f"Loading {model}...")

            # Load test metrics
            metrics = load_model_metrics(dataset_dir, model)

            if metrics is None:
                # Create placeholder entry for missing results
                metrics = {
                    'F1': 'Not Available',
                    'AUROC': 'Not Available',
                    'Accuracy': 'Not Available',
                    'Precision': 'Not Available',
                    'Recall': 'Not Available',
                    'Specificity': 'Not Available'
                }

            # Load best hyperparameters for context
            best_params = load_hyperparameter_summary(dataset_dir, model)

            # Create result entry
            result_entry = {
                'Model': model.upper(),
                'F1': metrics['F1'],
                'AUROC': metrics['AUROC'],
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'Specificity': metrics['Specificity'],
                'Best_Params': str(best_params) if best_params else 'N/A'
            }

            results[dataset_name].append(result_entry)

    # Convert to DataFrames
    ckd_df = pd.DataFrame(results['ckd'])
    mimic_df = pd.DataFrame(results['mimic'])

    return ckd_df, mimic_df


def format_metric_value(value, decimals=4):
    """
    Format metric values for display (handle both numeric and string values).

    Args:
        value: Metric value (float, int, or string)
        decimals (int): Number of decimal places

    Returns:
        str: Formatted value
    """
    if isinstance(value, (int, float)) and not np.isnan(value):
        return f"{value:.{decimals}f}"
    else:
        return str(value)


def create_summary_tables(ckd_df, mimic_df, save_dir="./results"):
    """
    Create and save summary tables in multiple formats.

    Args:
        ckd_df (DataFrame): CKD results
        mimic_df (DataFrame): MIMIC results
        save_dir (str): Directory to save results
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Format numeric columns for better display
    metric_columns = ['F1', 'AUROC', 'Accuracy', 'Precision', 'Recall', 'Specificity']

    for df in [ckd_df, mimic_df]:
        for col in metric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: format_metric_value(x) if x != 'Not Available' else 'N/A')

    # Save CSV files
    ckd_csv_path = os.path.join(save_dir, "summary_ckd_results.csv")
    mimic_csv_path = os.path.join(save_dir, "summary_mimic_results.csv")

    ckd_df.to_csv(ckd_csv_path, index=False)
    mimic_df.to_csv(mimic_csv_path, index=False)

    print(f"✅ CSV files saved:")
    print(f"   - {ckd_csv_path}")
    print(f"   - {mimic_csv_path}")

    # Save Excel files (if openpyxl is available)
    try:
        ckd_excel_path = os.path.join(save_dir, "summary_ckd_results.xlsx")
        mimic_excel_path = os.path.join(save_dir, "summary_mimic_results.xlsx")

        ckd_df.to_excel(ckd_excel_path, index=False, sheet_name='CKD_Results')
        mimic_df.to_excel(mimic_excel_path, index=False, sheet_name='MIMIC_Results')

        print(f"✅ Excel files saved:")
        print(f"   - {ckd_excel_path}")
        print(f"   - {mimic_excel_path}")

    except ImportError:
        print("⚠️  openpyxl not available - Excel files not created")

    # Create combined summary
    combined_path = os.path.join(save_dir, "combined_results_summary.xlsx")
    try:
        with pd.ExcelWriter(combined_path, engine='openpyxl') as writer:
            ckd_df.to_excel(writer, sheet_name='CKD_Dataset', index=False)
            mimic_df.to_excel(writer, sheet_name='MIMIC_Dataset', index=False)

        print(f"✅ Combined Excel file saved: {combined_path}")
    except ImportError:
        pass


def display_results_tables(ckd_df, mimic_df):
    """
    Display formatted results tables in console.

    Args:
        ckd_df (DataFrame): CKD results
        mimic_df (DataFrame): MIMIC results
    """
    print("\n" + "="*80)
    print("                        COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)

    # CKD Results Table
    print(f"\n{'CKD DATASET RESULTS':^80}")
    print("-"*80)

    if not ckd_df.empty:
        # Create display table without Best_Params for cleaner output
        display_ckd = ckd_df[['Model', 'F1', 'AUROC', 'Accuracy', 'Precision', 'Recall', 'Specificity']].copy()
        print(display_ckd.to_string(index=False, col_space=12))
    else:
        print("No CKD results found.")

    # MIMIC Results Table
    print(f"\n\n{'MIMIC-IV DATASET RESULTS':^80}")
    print("-"*80)

    if not mimic_df.empty:
        # Create display table without Best_Params for cleaner output
        display_mimic = mimic_df[['Model', 'F1', 'AUROC', 'Accuracy', 'Precision', 'Recall', 'Specificity']].copy()
        print(display_mimic.to_string(index=False, col_space=12))
    else:
        print("No MIMIC results found.")

    print("\n" + "="*80)


def generate_analysis_summary(ckd_df, mimic_df, save_dir="./results"):
    """
    Generate additional analysis and insights.

    Args:
        ckd_df (DataFrame): CKD results
        mimic_df (DataFrame): MIMIC results
        save_dir (str): Directory to save analysis
    """
    analysis = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'datasets_analyzed': ['CKD', 'MIMIC-IV'],
        'models_evaluated': ['BiLSTM', 'RETAIN', 'Transformer', 'AdaCare', 'StageNet', 'CoI'],
        'summary': {}
    }

    # Find best performing models per dataset
    for dataset_name, df in [('CKD', ckd_df), ('MIMIC-IV', mimic_df)]:
        if df.empty:
            continue

        # Convert numeric columns for analysis
        numeric_df = df.copy()
        for col in ['F1', 'AUROC', 'Accuracy', 'Precision']:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

        best_models = {}
        for metric in ['F1', 'AUROC', 'Accuracy', 'Precision']:
            if metric in numeric_df.columns:
                best_idx = numeric_df[metric].idxmax()
                if not np.isnan(numeric_df.loc[best_idx, metric]):
                    best_models[f'Best_{metric}'] = {
                        'model': numeric_df.loc[best_idx, 'Model'],
                        'value': float(numeric_df.loc[best_idx, metric])
                    }

        analysis['summary'][dataset_name] = best_models

    # Save analysis
    analysis_path = os.path.join(save_dir, "results_analysis_summary.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"✅ Analysis summary saved: {analysis_path}")


def main():
    """
    Main function to run the complete results analysis.
    """
    print("🔍 Chain-of-Dynamics Results Analysis")
    print("=====================================")

    # Collect all results
    print("\n📊 Collecting results from all models...")
    ckd_df, mimic_df = collect_all_results()

    # Display results in console
    display_results_tables(ckd_df, mimic_df)

    # Create and save summary tables
    print(f"\n💾 Saving results tables...")
    create_summary_tables(ckd_df, mimic_df)

    # Generate additional analysis
    print(f"\n📈 Generating analysis summary...")
    generate_analysis_summary(ckd_df, mimic_df)

    print(f"\n✅ Results analysis complete!")
    print(f"📁 All files saved to ./results/ directory")


if __name__ == "__main__":
    main()