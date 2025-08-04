import os
import yaml

def report_best_model(summary_path=None):
    if summary_path is None:
        # 默认查找 Outputs/coi_hypersearch_summary.yaml
        summary_path = os.path.join('Outputs', 'coi_hypersearch_summary.yaml')
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return
    with open(summary_path, 'r') as f:
        summary = yaml.safe_load(f)
    print("==== Best Model from Hyperparameter Search ====")
    print(f"Best F1: {summary.get('best_f1', 'N/A')}")
    print(f"Best Params: {summary.get('best_params', {})}")
    print(f"Model Path: {summary.get('model_path', 'N/A')}")
    print(f"Timestamp: {summary.get('timestamp', 'N/A')}")

if __name__ == "__main__":
    report_best_model() 