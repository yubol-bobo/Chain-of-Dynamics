# Data Directory

This directory contains instructions for obtaining the datasets used in this research.

## Datasets

### 1. Chronic Kidney Disease (CKD) Dataset
- **Type**: Private dataset (not publicly available)
- **Description**: Longitudinal clinical data from CKD patients with 24-month follow-up
- **Features**: 38 clinical features including demographics, comorbidities, laboratory biomarkers, and healthcare utilization
- **Target**: End-Stage Renal Disease (ESRD) progression prediction
- **Size**: 1,422 patients with 8 temporal observations per patient
- **Class distribution**: 6.0% ESRD progression rate

### 2. MIMIC-IV Dataset
- **Type**: Public dataset
- **Description**: Critical care database from Beth Israel Deaconess Medical Center
- **Version**: v3.1
- **Access**: Requires PhysioNet credentialed access
- **URL**: https://physionet.org/content/mimiciv/3.1/
- **Features**: 15 clinical features (vital signs, laboratory tests, demographics)
- **Target**: In-hospital mortality prediction
- **Size**: 65,366 patients with 48 hours of hourly measurements
- **Class distribution**: 10.8% mortality rate

## Data Structure

After obtaining the datasets, organize them as follows:

```
data/
????????? ckd_data/
???   ????????? raw/
???   ???   ????????? Agg_data_24.csv
???   ???   ????????? agr_longitudinal_df_24.csv
???   ???   ????????? final_df_24.csv
???   ???   ????????? lab_test_longitudinal_df_2009_24.csv
???   ???   ????????? longitudinal_comorbidity_df_2009_24.csv
???   ???   ????????? stage_merged_df_matched.csv
???   ????????? processed/
???       ????????? ckd_merged_data_for_modeling.csv
????????? mimic-iv-3.1/
    ????????? hosp/
    ????????? icu/
    ????????? processed/
        ????????? X_mimiciv_parallel.npy
        ????????? y_mimiciv_parallel.npy
```


## Preprocessing

### CKD Dataset
The CKD dataset should be preprocessed and saved as `ckd_merged_data_for_modeling.csv` with temporal features organized by month.

You can regenerate it from raw files with:

```bash
python scripts/preprocess_ckd_fast.py
```


### MIMIC-IV Dataset
Use the preprocessing script to convert raw MIMIC-IV data:

```bash
python scripts/preprocess_mimiciv.py
```

This script will:
1. Extract cohort of first ICU stays
2. Generate hourly features over 48 hours
3. Save processed data as NumPy arrays

## Privacy and Ethics

- The CKD dataset contains private patient information and cannot be shared
- MIMIC-IV data requires appropriate credentialed access and data use agreements
- All data should be handled according to institutional IRB requirements
- No actual patient data is included in this repository 