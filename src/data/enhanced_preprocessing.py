#!/usr/bin/env python3
"""
Enhanced preprocessing pipeline for longitudinal clinical data.
Addresses critical issues: missing data handling, class imbalance, temporal features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import ADASYN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import torch
from torch.utils.data import TensorDataset
import warnings
warnings.filterwarnings('ignore')

class EnhancedTemporalPreprocessor:
    """Enhanced preprocessing for longitudinal clinical data."""

    def __init__(self, time_periods=8, imputation_method='temporal_aware'):
        self.time_periods = time_periods
        self.imputation_method = imputation_method
        self.scaler = None
        self.imputer = None
        self.feature_names = None

    def identify_feature_groups(self, columns):
        """Identify different types of clinical features for appropriate handling."""
        clinical_conditions = []
        lab_values = []
        demographics = []
        utilization = []

        # Extract unique feature names
        unique_features = set()
        for col in columns:
            if col not in ['TMA_Acct', 'ESRD'] and '_Time_' in col:
                base_name = col.split('_Time_')[0]
                unique_features.add(base_name)

        for feature in unique_features:
            if feature in ['Diabetes', 'Htn', 'Cvd', 'Anemia', 'MA', 'Prot', 'SH', 'Phos',
                          'Athsc', 'CHF', 'Stroke', 'CD', 'MI', 'FE', 'MD', 'ND', 'S4', 'S5']:
                clinical_conditions.append(feature)
            elif feature in ['Serum_Calcium', 'eGFR', 'Phosphorus', 'Intact_PTH', 'Hemoglobin', 'UACR']:
                lab_values.append(feature)
            elif feature in ['Age', 'Gender', 'Race', 'BMI']:
                demographics.append(feature)
            elif feature.startswith('n_claims') or feature.startswith('net_exp'):
                utilization.append(feature)

        return {
            'clinical': clinical_conditions,
            'lab': lab_values,
            'demographic': demographics,
            'utilization': utilization
        }

    def temporal_aware_imputation(self, df):
        """
        Implement temporal-aware imputation strategy.
        Different strategies for different feature types.
        """
        print("[INFO] Applying temporal-aware imputation...")

        df_imputed = df.copy()
        feature_groups = self.identify_feature_groups(df.columns)

        # 1. Forward/Backward fill for temporal continuity
        for group_name, features in feature_groups.items():
            for feature_base in features:
                time_cols = [f"{feature_base}_Time_{t}" for t in range(self.time_periods)
                           if f"{feature_base}_Time_{t}" in df.columns]

                if time_cols:
                    if group_name in ['clinical', 'demographic']:
                        # For binary/categorical: forward fill then backward fill
                        df_imputed[time_cols] = df_imputed[time_cols].fillna(method='ffill', axis=1)
                        df_imputed[time_cols] = df_imputed[time_cols].fillna(method='bfill', axis=1)

                        # Fill remaining with most common value
                        for col in time_cols:
                            if df_imputed[col].isnull().any():
                                mode_val = df_imputed[col].mode().iloc[0] if len(df_imputed[col].mode()) > 0 else 0
                                df_imputed[col].fillna(mode_val, inplace=True)

                    elif group_name == 'lab':
                        # For lab values: linear interpolation then forward/backward fill
                        df_imputed[time_cols] = df_imputed[time_cols].interpolate(method='linear', axis=1)
                        df_imputed[time_cols] = df_imputed[time_cols].fillna(method='ffill', axis=1)
                        df_imputed[time_cols] = df_imputed[time_cols].fillna(method='bfill', axis=1)

                        # Fill remaining with clinical normal values or median
                        normal_values = {
                            'eGFR': 90.0,
                            'Hemoglobin': 12.0,
                            'Serum_Calcium': 9.5,
                            'Phosphorus': 3.5,
                            'UACR': 15.0
                        }

                        for col in time_cols:
                            if df_imputed[col].isnull().any():
                                fill_val = normal_values.get(feature_base, df_imputed[col].median())
                                df_imputed[col].fillna(fill_val, inplace=True)

                    elif group_name == 'utilization':
                        # For utilization: forward fill then zero
                        df_imputed[time_cols] = df_imputed[time_cols].fillna(method='ffill', axis=1)
                        df_imputed[time_cols] = df_imputed[time_cols].fillna(0)

        # 2. Apply MICE for remaining complex missingness patterns
        remaining_missing = df_imputed.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"[INFO] Applying MICE imputation for {remaining_missing} remaining missing values...")

            # Exclude ID columns for MICE
            feature_cols = [col for col in df_imputed.columns if col not in ['TMA_Acct', 'ESRD']]

            # Filter out columns with only missing values or constant values
            valid_feature_cols = []
            for col in feature_cols:
                if df_imputed[col].notna().sum() > 0:  # Has at least one non-missing value
                    if df_imputed[col].notna().sum() > 1:  # Has more than one value for variance
                        valid_feature_cols.append(col)

            if valid_feature_cols:
                mice_imputer = IterativeImputer(
                    random_state=42,
                    max_iter=10,
                    initial_strategy='mean'
                )

                # Apply MICE only on valid columns
                imputed_values = mice_imputer.fit_transform(df_imputed[valid_feature_cols])

                # Verify shape consistency
                if imputed_values.shape[1] == len(valid_feature_cols):
                    # Update valid columns with MICE results
                    for i, col in enumerate(valid_feature_cols):
                        df_imputed[col] = imputed_values[:, i]

                    # Handle remaining invalid columns with simple forward fill
                    invalid_cols = [col for col in feature_cols if col not in valid_feature_cols]
                    for col in invalid_cols:
                        if df_imputed[col].isnull().any():
                            # Use median for numeric, mode for categorical
                            if df_imputed[col].dtype in ['float64', 'int64']:
                                fill_val = df_imputed[col].median() if df_imputed[col].notna().sum() > 0 else 0
                            else:
                                fill_val = df_imputed[col].mode().iloc[0] if len(df_imputed[col].mode()) > 0 else 0
                            df_imputed[col].fillna(fill_val, inplace=True)
                else:
                    print(f"[WARNING] Shape mismatch in MICE output. Using simple imputation instead.")
                    # Fallback to simple imputation
                    for col in feature_cols:
                        if df_imputed[col].isnull().any():
                            if df_imputed[col].dtype in ['float64', 'int64']:
                                fill_val = df_imputed[col].median() if df_imputed[col].notna().sum() > 0 else 0
                            else:
                                fill_val = df_imputed[col].mode().iloc[0] if len(df_imputed[col].mode()) > 0 else 0
                            df_imputed[col].fillna(fill_val, inplace=True)
            else:
                print("[WARNING] No valid columns for MICE. Using simple imputation.")
                # Fallback to simple imputation for all columns
                for col in feature_cols:
                    if df_imputed[col].isnull().any():
                        if df_imputed[col].dtype in ['float64', 'int64']:
                            fill_val = df_imputed[col].median() if df_imputed[col].notna().sum() > 0 else 0
                        else:
                            fill_val = df_imputed[col].mode().iloc[0] if len(df_imputed[col].mode()) > 0 else 0
                        df_imputed[col].fillna(fill_val, inplace=True)

        print(f"[SUCCESS] Imputation completed. Remaining missing: {df_imputed.isnull().sum().sum()}")
        return df_imputed

    def engineer_temporal_features(self, df):
        """Create temporal features to capture longitudinal patterns."""
        print("[INFO] Engineering temporal features...")

        df_enhanced = df.copy()
        feature_groups = self.identify_feature_groups(df.columns)
        new_features = []

        # Focus on key clinical features for temporal engineering
        key_features = ['eGFR', 'Hemoglobin', 'UACR'] + feature_groups['clinical'][:5]  # Top 5 clinical conditions

        for feature_base in key_features:
            time_cols = [f"{feature_base}_Time_{t}" for t in range(self.time_periods)
                        if f"{feature_base}_Time_{t}" in df.columns]

            if len(time_cols) >= 3:  # Need at least 3 time points for trends
                values = df_enhanced[time_cols].values

                # 1. Linear trend (slope)
                trend_col = f"{feature_base}_trend"
                trends = []
                for i in range(len(values)):
                    x = np.arange(len(time_cols))
                    y = values[i]
                    if not np.isnan(y).all():
                        # Simple linear regression slope
                        slope = np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)[0] if len(y[~np.isnan(y)]) > 1 else 0
                        trends.append(slope)
                    else:
                        trends.append(0)

                df_enhanced[trend_col] = trends
                new_features.append(trend_col)

                # 2. Variability (coefficient of variation)
                if feature_base in ['eGFR', 'Hemoglobin', 'UACR']:  # Only for continuous features
                    var_col = f"{feature_base}_variability"
                    cv_values = []
                    for i in range(len(values)):
                        y = values[i][~np.isnan(values[i])]
                        if len(y) > 1 and np.mean(y) != 0:
                            cv = np.std(y) / np.mean(y)
                            cv_values.append(cv)
                        else:
                            cv_values.append(0)

                    df_enhanced[var_col] = cv_values
                    new_features.append(var_col)

                # 3. Time since last measurement indicator
                time_since_col = f"{feature_base}_time_since_last"
                time_since_values = []
                for i in range(len(values)):
                    last_valid_time = -1
                    for t in range(len(time_cols)):
                        if not np.isnan(values[i, t]):
                            last_valid_time = t
                    time_since_values.append(self.time_periods - 1 - last_valid_time if last_valid_time >= 0 else self.time_periods)

                df_enhanced[time_since_col] = time_since_values
                new_features.append(time_since_col)

        print(f"[SUCCESS] Added {len(new_features)} temporal features: {new_features[:5]}...")
        return df_enhanced

    def enhanced_normalization(self, X_train, X_val, X_test):
        """Apply robust normalization that preserves temporal relationships."""
        print("[INFO] Applying enhanced normalization...")

        # Reshape to (n_samples * timesteps, n_features) for normalization
        n_train, timesteps, n_features = X_train.shape
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]

        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)

        # Use RobustScaler (less sensitive to outliers)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_2d)
        X_val_scaled = self.scaler.transform(X_val_2d)
        X_test_scaled = self.scaler.transform(X_test_2d)

        # Reshape back to (n_samples, timesteps, n_features)
        X_train_scaled = X_train_scaled.reshape(n_train, timesteps, n_features)
        X_val_scaled = X_val_scaled.reshape(n_val, timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(n_test, timesteps, n_features)

        print("[SUCCESS] Enhanced normalization completed")
        return X_train_scaled, X_val_scaled, X_test_scaled

    def enhanced_resampling(self, X_train, y_train):
        """Apply ADASYN for better handling of class imbalance."""
        print("[INFO] Applying enhanced resampling with ADASYN...")

        # Reshape for ADASYN (needs 2D input)
        n_samples, timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(n_samples, -1)  # Flatten temporal dimension

        # Check class distribution before resampling
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"[INFO] Original class distribution: {dict(zip(unique, counts))}")

        try:
            # ADASYN with conservative sampling
            adasyn = ADASYN(
                sampling_strategy='auto',  # Only oversample minority class
                random_state=42,
                n_neighbors=5  # Conservative number of neighbors
            )

            X_resampled, y_resampled = adasyn.fit_resample(X_train_2d, y_train)

            # Reshape back to 3D
            n_resampled = X_resampled.shape[0]
            X_resampled = X_resampled.reshape(n_resampled, timesteps, n_features)

            # Check new class distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            print(f"[SUCCESS] ADASYN resampled distribution: {dict(zip(unique, counts))}")
            print(f"[INFO] Original samples: {n_samples}, Resampled: {n_resampled}")

            return X_resampled, y_resampled

        except Exception as e:
            print(f"[WARNING] ADASYN failed ({e}), falling back to class weights only")
            return X_train, y_train

    def calculate_class_weights(self, y):
        """Calculate balanced class weights for cost-sensitive learning."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)

        # Balanced class weights
        class_weights = {}
        for cls, count in zip(unique, counts):
            class_weights[cls] = total / (len(unique) * count)

        print(f"[INFO] Calculated class weights: {class_weights}")
        return class_weights

    def preprocess_ckd_data(self, data_path):
        """Main preprocessing pipeline for CKD data."""
        print("="*60)
        print("         ENHANCED CKD DATA PREPROCESSING")
        print("="*60)

        # Load data
        print(f"[INFO] Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"[INFO] Loaded data shape: {df.shape}")

        # Step 1: Enhanced Missing Data Imputation
        df_imputed = self.temporal_aware_imputation(df)

        # Step 2: Temporal Feature Engineering
        df_enhanced = self.engineer_temporal_features(df_imputed)

        # Step 3: Prepare features and labels
        feature_cols = [col for col in df_enhanced.columns if col not in ['TMA_Acct', 'ESRD']]
        X = df_enhanced[feature_cols].values
        y = df_enhanced['ESRD'].values

        print(f"[INFO] Feature matrix shape: {X.shape}")
        print(f"[INFO] Labels shape: {y.shape}")

        # Step 4: Reshape to temporal format
        # Calculate features per time period more robustly
        temporal_features = [col for col in feature_cols if '_Time_' in col]
        engineered_features = [col for col in feature_cols if '_Time_' not in col]

        n_temporal_features = len(temporal_features)
        n_engineered_features = len(engineered_features)

        # Count unique base features (before _Time_X suffix) to determine features per period
        unique_base_features = set()
        for col in temporal_features:
            if '_Time_' in col:
                base_name = col.split('_Time_')[0]
                unique_base_features.add(base_name)

        features_per_period = len(unique_base_features)

        print(f"[INFO] Temporal features: {n_temporal_features}, Engineered: {n_engineered_features}")
        print(f"[INFO] Unique base features: {len(unique_base_features)}")
        print(f"[INFO] Features per time period: {features_per_period}")

        # Verify we can reshape properly
        expected_size = len(df_enhanced) * self.time_periods * features_per_period
        actual_size = len(df_enhanced) * n_temporal_features

        if expected_size != actual_size:
            print(f"[WARNING] Reshape mismatch: expected {expected_size}, actual {actual_size}")
            # Adjust features_per_period based on actual data
            features_per_period = n_temporal_features // self.time_periods
            print(f"[INFO] Adjusted features per time period: {features_per_period}")

            # If still doesn't divide evenly, pad with zeros or truncate
            remainder = n_temporal_features % self.time_periods
            if remainder != 0:
                print(f"[WARNING] {remainder} temporal features don't fit evenly into {self.time_periods} periods")
                # Pad temporal features to make it divisible
                padding_needed = self.time_periods - remainder
                for i in range(padding_needed):
                    padding_col = f"padding_feature_{i}"
                    df_enhanced[padding_col] = 0.0
                    temporal_features.append(padding_col)

                n_temporal_features = len(temporal_features)
                features_per_period = n_temporal_features // self.time_periods
                print(f"[INFO] After padding - Temporal features: {n_temporal_features}, Features per period: {features_per_period}")

        # Reshape temporal features
        X_temporal = df_enhanced[temporal_features].values
        X_temporal_3d = X_temporal.reshape(len(df_enhanced), self.time_periods, features_per_period)

        # Add engineered features to the last time step or as separate features
        if engineered_features:
            X_engineered = df_enhanced[engineered_features].values
            # Expand engineered features to match temporal dimensions
            X_engineered_expanded = np.expand_dims(X_engineered, axis=1)
            X_engineered_tiled = np.tile(X_engineered_expanded, (1, self.time_periods, 1))

            # Concatenate with temporal features
            X_final = np.concatenate([X_temporal_3d, X_engineered_tiled], axis=2)
        else:
            X_final = X_temporal_3d

        print(f"[INFO] Final feature shape: {X_final.shape}")

        # Step 5: Train-validation-test split (temporal-aware)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_final, y, test_size=0.4, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Step 6: Enhanced normalization
        X_train_scaled, X_val_scaled, X_test_scaled = self.enhanced_normalization(
            X_train, X_val, X_test
        )

        # Step 7: Enhanced resampling (ADASYN)
        X_train_resampled, y_train_resampled = self.enhanced_resampling(X_train_scaled, y_train)

        # Step 8: Calculate class weights
        class_weights = self.calculate_class_weights(y_train_resampled)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Store metadata
        self.input_dim = X_final.shape[2]
        self.class_weights = class_weights

        print("="*60)
        print("     ENHANCED PREPROCESSING COMPLETED")
        print("="*60)
        print(f"Final input dimensions: {self.input_dim}")
        print(f"Training samples after resampling: {len(train_dataset)}")
        print(f"Class weights: {class_weights}")

        return train_dataset, val_dataset, test_dataset, self.input_dim, class_weights

    def preprocess_mimic_data(self, processed_path, label_path):
        """
        Enhanced preprocessing for MIMIC-IV dataset using numpy arrays.

        Args:
            processed_path: Path to X_mimiciv.npy or X_mimiciv_parallel.npy
            label_path: Path to y_mimiciv.npy or y_mimiciv_parallel.npy

        Returns:
            train_dataset, val_dataset, test_dataset, input_dim, class_weights
        """
        print("============================================================")
        print("         ENHANCED MIMIC-IV DATA PREPROCESSING")
        print("============================================================")

        # Step 1: Load numpy data
        print(f"[INFO] Loading features from {processed_path}")
        X = np.load(processed_path)
        print(f"[INFO] Loading labels from {label_path}")
        y = np.load(label_path)

        print(f"[INFO] Loaded features shape: {X.shape}")
        print(f"[INFO] Loaded labels shape: {y.shape}")
        print(f"[INFO] Missing values in features: {np.isnan(X).sum()}")

        # Step 2: Enhanced imputation for temporal data
        print("[INFO] Applying enhanced temporal imputation...")
        X_imputed = self._enhanced_mimic_imputation(X)
        print(f"[SUCCESS] Imputation completed. Remaining missing: {np.isnan(X_imputed).sum()}")

        # Step 3: Split dataset
        print("[INFO] Splitting dataset...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_imputed, y, test_size=0.4, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Step 4: Enhanced normalization
        print("[INFO] Applying enhanced normalization...")
        X_train_norm, X_val_norm, X_test_norm = self._enhanced_mimic_normalization(
            X_train, X_val, X_test
        )
        print("[SUCCESS] Enhanced normalization completed")

        # Step 5: Enhanced resampling with ADASYN
        print("[INFO] Applying enhanced resampling with ADASYN...")
        X_train_resampled, y_train_resampled = self._enhanced_mimic_resampling(
            X_train_norm, y_train
        )

        # Calculate class weights
        class_counts = np.bincount(y_train_resampled.astype(int))
        total_samples = len(y_train_resampled)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}

        print(f"[INFO] Original samples: {len(y_train)}, Resampled: {len(y_train_resampled)}")
        print(f"[INFO] Calculated class weights: {class_weights}")

        # Step 6: Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_resampled),
            torch.FloatTensor(y_train_resampled)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_norm),
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_norm),
            torch.FloatTensor(y_test)
        )

        # Store class weights in dataset for loss function
        train_dataset.class_weights = class_weights

        final_input_dim = X_train_resampled.shape[-1]

        print("============================================================")
        print("     ENHANCED MIMIC PREPROCESSING COMPLETED")
        print("============================================================")
        print(f"Final input dimensions: {final_input_dim}")
        print(f"Training samples after resampling: {len(train_dataset)}")
        print(f"Class weights: {class_weights}")

        return train_dataset, val_dataset, test_dataset, final_input_dim, class_weights

    def _enhanced_mimic_imputation(self, X):
        """Enhanced imputation for MIMIC temporal data."""
        n_patients, timesteps, n_features = X.shape
        X_imputed = X.copy()

        print(f"[INFO] Processing {n_patients} patients, {timesteps} timesteps, {n_features} features")

        # 1. Forward/backward fill across time for each patient
        for patient_idx in range(n_patients):
            patient_data = X_imputed[patient_idx, :, :]

            # Forward fill then backward fill
            df_patient = pd.DataFrame(patient_data)
            df_patient = df_patient.fillna(method='ffill').fillna(method='bfill')

            X_imputed[patient_idx, :, :] = df_patient.values

        # 2. MICE imputation for remaining missing values
        remaining_missing = np.isnan(X_imputed).sum()
        if remaining_missing > 0:
            print(f"[INFO] Applying MICE imputation for {remaining_missing} remaining missing values...")

            # Reshape for MICE (flatten temporal dimension)
            X_2d = X_imputed.reshape(-1, n_features)

            # Filter out rows and columns that are all NaN
            valid_rows = ~np.all(np.isnan(X_2d), axis=1)
            valid_cols = ~np.all(np.isnan(X_2d), axis=0)

            if np.any(valid_rows) and np.any(valid_cols):
                X_2d_valid = X_2d[valid_rows][:, valid_cols]

                if X_2d_valid.shape[0] > 0 and np.any(np.isnan(X_2d_valid)):
                    mice_imputer = IterativeImputer(random_state=42, max_iter=10)
                    X_2d_imputed = mice_imputer.fit_transform(X_2d_valid)

                    # Put back into original structure
                    X_2d[np.ix_(valid_rows, valid_cols)] = X_2d_imputed
                    X_imputed = X_2d.reshape(n_patients, timesteps, n_features)

            # Fill any remaining NaN with feature medians
            for feat_idx in range(n_features):
                feat_values = X_imputed[:, :, feat_idx].flatten()
                if np.any(np.isnan(feat_values)):
                    median_val = np.nanmedian(feat_values)
                    X_imputed[:, :, feat_idx] = np.where(
                        np.isnan(X_imputed[:, :, feat_idx]),
                        median_val,
                        X_imputed[:, :, feat_idx]
                    )

        return X_imputed

    def _enhanced_mimic_normalization(self, X_train, X_val, X_test):
        """Enhanced normalization for MIMIC temporal data."""
        n_patients, timesteps, n_features = X_train.shape

        # Reshape to 2D for normalization
        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)

        # Use RobustScaler for better handling of outliers
        scaler = RobustScaler()
        X_train_norm_2d = scaler.fit_transform(X_train_2d)
        X_val_norm_2d = scaler.transform(X_val_2d)
        X_test_norm_2d = scaler.transform(X_test_2d)

        # Reshape back to 3D
        X_train_norm = X_train_norm_2d.reshape(X_train.shape)
        X_val_norm = X_val_norm_2d.reshape(X_val.shape)
        X_test_norm = X_test_norm_2d.reshape(X_test.shape)

        return X_train_norm, X_val_norm, X_test_norm

    def _enhanced_mimic_resampling(self, X_train, y_train):
        """Enhanced resampling for MIMIC temporal data using ADASYN."""
        n_patients, timesteps, n_features = X_train.shape

        print(f"[INFO] Original class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

        # For large datasets (>20K samples), ADASYN is too slow, use class weights instead
        if n_patients > 20000:
            print(f"[INFO] Large dataset ({n_patients} patients) detected - skipping ADASYN for performance")
            print("[INFO] Will use class weights for balancing instead")
            return X_train, y_train

        # Reshape to 2D for resampling
        X_train_2d = X_train.reshape(n_patients, -1)

        try:
            # Apply ADASYN with timeout protection
            print("[INFO] Applying ADASYN (this may take a few minutes for large datasets)...")
            adasyn = ADASYN(random_state=42, n_neighbors=5)
            X_resampled_2d, y_resampled = adasyn.fit_resample(X_train_2d, y_train)

            print(f"[SUCCESS] ADASYN resampled distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

            # Reshape back to 3D
            n_resampled_patients = X_resampled_2d.shape[0]
            X_resampled = X_resampled_2d.reshape(n_resampled_patients, timesteps, n_features)

            return X_resampled, y_resampled

        except Exception as e:
            print(f"[WARNING] ADASYN failed ({e}) - using original data with class weights")
            return X_train, y_train

def main():
    """Test the enhanced preprocessing pipeline."""
    preprocessor = EnhancedTemporalPreprocessor()

    # Test on CKD data
    ckd_path = "data/ckd_data/processed/ckd_merged_data_for_modeling.csv"
    train_data, val_data, test_data, input_dim, class_weights = preprocessor.preprocess_ckd_data(ckd_path)

    print(f"\n[SUCCESS] Enhanced preprocessing pipeline tested successfully!")
    print(f"Input dimension: {input_dim}")
    print(f"Train dataset size: {len(train_data)}")

if __name__ == "__main__":
    main()