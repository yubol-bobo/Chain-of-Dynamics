#!/usr/bin/env python3
"""
Python script to run all Chain-of-Dynamics experiments automatically.
This is more reliable than batch files and works cross-platform.

Usage examples:
    # Quick testing with 2 combinations
    python run_all_experiments.py --combinations 2

    # Test specific models and datasets
    python run_all_experiments.py -c 5 --models bilstm retain --datasets ckd

    # Full experiment (default)
    python run_all_experiments.py

    # Custom conda environment
    python run_all_experiments.py --env my_env --combinations 10
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"Command: {' '.join(command)}")
    print('='*80)

    try:
        start_time = time.time()
        result = subprocess.run(command, check=True, capture_output=False)
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n[SUCCESS] {description} completed in {duration:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {description} failed with error: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run all Chain-of-Dynamics experiments with enhanced preprocessing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--combinations', '-c',
        type=int,
        default=500,
        help='Number of hyperparameter combinations to test per model (use smaller numbers like 2-10 for quick testing)'
    )

    parser.add_argument(
        '--env', '-e',
        type=str,
        default='coi',
        help='Conda environment name to use'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['ckd', 'mimic'],
        choices=['ckd', 'mimic'],
        help='Datasets to run experiments on'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['bilstm', 'retain', 'transformer', 'adacare', 'stagenet', 'coi'],
        choices=['bilstm', 'retain', 'transformer', 'adacare', 'stagenet', 'coi'],
        help='Models to train'
    )

    args = parser.parse_args()

    combinations = args.combinations
    conda_env = args.env

    print("="*80)
    print("              CHAIN-OF-DYNAMICS EXPERIMENT PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Combinations per model: {combinations}")
    print(f"  - Conda environment: {conda_env}")
    print(f"  - Enhanced preprocessing: ENABLED")
    print(f"  - Datasets: {', '.join(args.datasets)}")
    print(f"  - Models: {', '.join(args.models)}")
    print(f"  - Working directory: {os.getcwd()}")
    print()

    # Show helpful usage examples
    if combinations <= 10:
        print("🧪 TESTING MODE: Using small number of combinations for quick testing")
    else:
        print("🚀 FULL EXPERIMENT MODE: Running comprehensive hyperparameter search")
    print()

    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("[INFO] Experiment cancelled by user")
        return

    # Create results directories
    os.makedirs("results/ckd", exist_ok=True)
    os.makedirs("results/mimic", exist_ok=True)

    # Use models and datasets from command line arguments
    models = args.models
    datasets = args.datasets

    total_experiments = len(models) * len(datasets)
    current_experiment = 0
    failed_experiments = []
    completed_experiments = []

    print(f"\n[INFO] Starting pipeline with {total_experiments} experiments")
    pipeline_start = time.time()

    # Run all experiments
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"TRAINING ALL MODELS ON {dataset.upper()} DATASET")
        print('='*60)

        for model in models:
            current_experiment += 1

            description = f"Training {model.upper()} on {dataset.upper()} [{current_experiment}/{total_experiments}]"

            # Build command with enhanced preprocessing
            command = [
                'conda', 'run', '-n', conda_env, 'python',
                'scripts/train.py',
                '--model', model,
                '--dataset', dataset,
                '--enhanced',  # Enable enhanced preprocessing by default
                '--hyperparameter-search',
                '--n-combinations', str(combinations)
            ]

            # Run the training
            success = run_command(command, description)

            experiment_name = f"{model}-{dataset}"
            if success:
                completed_experiments.append(experiment_name)
            else:
                failed_experiments.append(experiment_name)

            # Show detailed progress
            progress = (current_experiment * 100) // total_experiments
            completed_count = len(completed_experiments)
            failed_count = len(failed_experiments)

            print(f"\n[PROGRESS] {current_experiment}/{total_experiments} ({progress}%) completed")
            print(f"           ✅ Successful: {completed_count} | ❌ Failed: {failed_count}")

            if completed_experiments:
                print(f"           Completed: {', '.join(completed_experiments)}")

            if failed_experiments:
                print(f"           Failed: {', '.join(failed_experiments)}")

    # Training completion summary
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*80}")
    print("                     TRAINING COMPLETED")
    print('='*80)

    if not failed_experiments:
        print("[SUCCESS] All experiments completed successfully!")
    else:
        print(f"[WARNING] {len(failed_experiments)} experiments failed:")
        for failed in failed_experiments:
            print(f"  - {failed}")

    print(f"Total pipeline time: {hours}h {minutes}m {seconds}s")
    print(f"Successful experiments: {total_experiments - len(failed_experiments)}/{total_experiments}")

    # Generate results summary
    print(f"\n{'='*80}")
    print("                  GENERATING RESULTS SUMMARY")
    print('='*80)

    analysis_command = ['conda', 'run', '-n', conda_env, 'python', 'scripts/analyze_results.py']
    analysis_success = run_command(analysis_command, "Generating results analysis")

    if analysis_success:
        print("\n[SUCCESS] Results analysis completed!")
        print("\nGenerated files:")

        result_files = [
            "results/summary_ckd_results.csv",
            "results/summary_mimic_results.csv",
            "results/combined_results_summary.xlsx",
            "results/results_analysis_summary.json"
        ]

        for file_path in result_files:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
    else:
        print("\n[ERROR] Results analysis failed")

    print(f"\n{'='*80}")
    print("                    PIPELINE COMPLETED")
    print('='*80)
    print("[SUCCESS] Complete experiment pipeline finished!")
    print("Check the results/ directory for summary tables and individual model results.")

if __name__ == "__main__":
    main()