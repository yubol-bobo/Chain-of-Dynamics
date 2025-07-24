# Chain-of-Dynamics: Training & Analysis Guide

## 🚀 Training Models (with Robust Hyperparameter Saving)

All training (including hyperparameter search) now saves both the model weights and the hyperparameters used for training in the checkpoint file. This ensures robust, error-free analysis later.

### **Train a Model (with Hyperparameter Search)**

Example for RETAIN (similarly for other models):
```bash
python utils/train.py --model retain --hyperparameter-search --n-combinations 30
```
- The best model and its hyperparameters will be saved to:
  - `Outputs/saved_models/retain_best_model_hypersearch.pt`
  - `Outputs/saved_models/retain_hypersearch_summary.yaml`

#### **About `--model` and `--config`**
- `--model` is **required** and determines which model/config to use (e.g., `retain`, `tfcam`, `hcta`, etc.).
- `--config` is **optional**. If not provided, the script will automatically use `config/{model}_config.yaml`.
- You can still provide a custom config with `--config` if needed.

### **What’s Saved in the Checkpoint?**
- `model_state_dict`: The model weights
- `hyperparams`: The exact hyperparameters used for training

---

## 📊 Unified Model Analysis

You can analyze any trained model (RETAIN, TFCAM, HCTA, etc.) with a single command. The script will automatically:
- Load the correct config file
- Load the correct checkpoint
- Override config with the best hyperparameters (from summary or checkpoint)
- Run temporal, feature, and cross-temporal-feature analysis (if available)

### **Run Analysis**

Example for RETAIN:
```bash
python analysis/model_analysis.py --model retain --output visualizations/unified_analysis_retain
```

- For other models, just change `--model` (e.g., `tfcam`, `hcta`, `enhanced_tfcam`, `mstca`, `ctga`).
- The script will auto-detect the correct config and checkpoint.
- All results will be saved in the specified output directory.

### **How It Works**
- The script loads the best hyperparameters from `Outputs/saved_models/{model}_hypersearch_summary.yaml` and/or from the checkpoint itself.
- The model is always instantiated with the correct architecture, so you never get shape mismatch errors.
- Attention plots and analysis are saved in the output directory.

---

## 🛠️ Troubleshooting
- If you see a shape mismatch error, make sure you retrain your models with the latest code (so the checkpoint includes hyperparameters).
- If you want to analyze a model trained with old code, manually update the summary YAML to match the checkpoint, or retrain.

---

## 📚 Example Workflow
1. **Train with hyperparameter search:**
   ```bash
   python utils/train.py --model retain --hyperparameter-search --n-combinations 30
   ```
2. **Analyze the best model:**
   ```bash
   python analysis/model_analysis.py --model retain --output visualizations/unified_analysis_retain
   ```
3. **Repeat for other models as needed.**

---

For more details, see the code comments or ask for help!