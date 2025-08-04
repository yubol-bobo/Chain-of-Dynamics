# --- LOGGING & PARALLELIZATION IMPROVEMENT SUGGESTIONS ---
# 1. This script now uses batch processing to avoid memory bottlenecks.
# 2. Each batch only loads relevant chartevents/labevents for the patients in that batch.
# 3. You can increase BATCH_SIZE if you have more memory, or decrease if you run into issues.
# 4. For further speedup, you can add joblib.Parallel inside each batch if needed.
#
# Below: Added detailed logging to help pinpoint bottlenecks.

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

BATCH_SIZE = 1000

# Paths
ICU_PATH = "mimic-iv-3.1/icu/"
HOSP_PATH = "mimic-iv-3.1/hosp/"
OUTPUT_PATH = "mimic-iv-3.1/processed/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("[LOG] Loading icustays.csv.gz...")
t0 = time.time()
icustays = pd.read_csv(os.path.join(ICU_PATH, "icustays.csv.gz"))
print(f"[LOG] Loaded icustays.csv.gz in {time.time() - t0:.2f}s")

print("[LOG] Loading admissions.csv.gz...")
t0 = time.time()
admissions = pd.read_csv(os.path.join(HOSP_PATH, "admissions.csv.gz"))
print(f"[LOG] Loaded admissions.csv.gz in {time.time() - t0:.2f}s")

print("[LOG] Loading patients.csv.gz...")
t0 = time.time()
patients = pd.read_csv(os.path.join(HOSP_PATH, "patients.csv.gz"))
print(f"[LOG] Loaded patients.csv.gz in {time.time() - t0:.2f}s")

# 2. Select first ICU stay per patient
icustays = icustays.sort_values(["subject_id", "intime"])
first_icustays = icustays.groupby("subject_id").first().reset_index()

# 3. Merge with admissions and patients for label and demographics
cohort = first_icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left")
cohort = cohort.merge(patients, on="subject_id", how="left")

# 4. Define label: in-hospital mortality
cohort["label"] = cohort["hospital_expire_flag"]

# 5. Calculate age at admission
cohort["admittime"] = pd.to_datetime(cohort["admittime"])
cohort["anchor_year"] = cohort["anchor_year"].astype(int)
cohort["anchor_age"] = cohort["anchor_age"].astype(int)
cohort["age"] = cohort["anchor_age"] + (cohort["admittime"].dt.year - cohort["anchor_year"])
cohort["age"] = cohort["age"].clip(0, 120)  # Remove outliers

# 6. Prepare time series extraction
vital_signs = {
    "HeartRate": [220045],
    "SysBP": [220179],
    "DiasBP": [220180],
    "MeanBP": [220181],
    "RespRate": [220210],
    "TempC": [223761, 678],  # Celsius and Fahrenheit
    "SpO2": [220277],
}
lab_tests = {
    "Creatinine": [50912],
    "Glucose": [50931],
    "Sodium": [50983],
    "Potassium": [50971],
    "Hematocrit": [51221],
    "WBC": [51301],
}

# 7. Filter chartevents and labevents for relevant stays/subjects
filtered_chartevents_path = os.path.join(ICU_PATH, "chartevents_filtered.csv.gz")
filtered_labevents_path = os.path.join(HOSP_PATH, "labevents_filtered.csv.gz")

if not os.path.exists(filtered_chartevents_path):
    print("Filtering chartevents.csv.gz for relevant stay_ids...")
    stay_ids = set(cohort['stay_id'])
    filtered_chunks = []
    for chunk in pd.read_csv(os.path.join(ICU_PATH, "chartevents.csv.gz"), chunksize=10**6):
        filtered_chunk = chunk[chunk['stay_id'].isin(stay_ids)]
        filtered_chunks.append(filtered_chunk)
    filtered_chartevents = pd.concat(filtered_chunks)
    filtered_chartevents.to_csv(filtered_chartevents_path, index=False, compression='gzip')
    print(f"Saved filtered chartevents to {filtered_chartevents_path}")
else:
    print("Filtered chartevents already exists.")

if not os.path.exists(filtered_labevents_path):
    print("Filtering labevents.csv.gz for relevant subject_ids and hadm_ids...")
    subject_ids = set(cohort['subject_id'])
    hadm_ids = set(cohort['hadm_id'])
    filtered_chunks = []
    for chunk in pd.read_csv(os.path.join(HOSP_PATH, "labevents.csv.gz"), chunksize=10**6):
        filtered_chunk = chunk[(chunk['subject_id'].isin(subject_ids)) & (chunk['hadm_id'].isin(hadm_ids))]
        filtered_chunks.append(filtered_chunk)
    filtered_labevents = pd.concat(filtered_chunks)
    filtered_labevents.to_csv(filtered_labevents_path, index=False, compression='gzip')
    print(f"Saved filtered labevents to {filtered_labevents_path}")
else:
    print("Filtered labevents already exists.")

# 8. Load filtered data into memory
print("[LOG] Loading filtered chartevents into memory...")
t0 = time.time()
chartevents = pd.read_csv(filtered_chartevents_path)
print(f"[LOG] Loaded chartevents in {time.time() - t0:.2f}s, shape: {chartevents.shape}")

print("[LOG] Loading filtered labevents into memory...")
t0 = time.time()
labevents = pd.read_csv(filtered_labevents_path)
print(f"[LOG] Loaded labevents in {time.time() - t0:.2f}s, shape: {labevents.shape}")

window_hours = 48
n_features = len(vital_signs) + len(lab_tests) + 2  # +2 for gender, age

# Updated process_patient to accept batch-specific chartevents and labevents

def process_patient(row, chartevents_batch, labevents_batch):
    subject_id = row["subject_id"]
    stay_id = row["stay_id"]
    hadm_id = row["hadm_id"]
    intime = pd.to_datetime(row["intime"])
    gender = row["gender"]
    age = row["age"]
    label = row["label"]

    ts = np.full((window_hours, n_features), np.nan)
    gender_val = 1 if gender == "F" else 0
    age_val = age

    # Vitals
    ce = chartevents_batch[(chartevents_batch['stay_id'] == stay_id)]
    for i, (feat, itemids) in enumerate(vital_signs.items()):
        feat_data = ce[ce['itemid'].isin(itemids)]
        if not feat_data.empty:
            feat_data = feat_data.copy()
            feat_data['charttime'] = pd.to_datetime(feat_data['charttime'])
            feat_data = feat_data[(feat_data['charttime'] >= intime) & (feat_data['charttime'] < intime + pd.Timedelta(hours=window_hours))]
            for _, row2 in feat_data.iterrows():
                hour = int((row2['charttime'] - intime).total_seconds() // 3600)
                if 0 <= hour < window_hours:
                    ts[hour, i] = row2['valuenum']
            if feat == "TempC":
                ts[:, i] = np.where(ts[:, i] > 50, (ts[:, i] - 32) * 5.0/9.0, ts[:, i])

    # Labs
    le = labevents_batch[(labevents_batch['subject_id'] == subject_id) & (labevents_batch['hadm_id'] == hadm_id)]
    for i, (feat, itemids) in enumerate(lab_tests.items()):
        feat_data = le[le['itemid'].isin(itemids)]
        if not feat_data.empty:
            feat_data = feat_data.copy()
            feat_data['charttime'] = pd.to_datetime(feat_data['charttime'])
            feat_data = feat_data[(feat_data['charttime'] >= intime) & (feat_data['charttime'] < intime + pd.Timedelta(hours=window_hours))]
            for _, row2 in feat_data.iterrows():
                hour = int((row2['charttime'] - intime).total_seconds() // 3600)
                if 0 <= hour < window_hours:
                    ts[hour, len(vital_signs) + i] = row2['valuenum']

    # Add gender and age as static features
    ts[:, -2] = gender_val
    ts[:, -1] = age_val

    return ts, label

# Batch processing main loop
all_series = []
all_labels = []

print("[LOG] Starting batch processing...")
t0_total = time.time()
for batch_start in tqdm(range(0, len(cohort), BATCH_SIZE)):
    batch_end = min(batch_start + BATCH_SIZE, len(cohort))
    cohort_batch = cohort.iloc[batch_start:batch_end]
    print(f"[LOG] Processing batch {batch_start} to {batch_end}...")
    t0 = time.time()
    stay_ids = set(cohort_batch['stay_id'])
    subject_ids = set(cohort_batch['subject_id'])
    hadm_ids = set(cohort_batch['hadm_id'])
    chartevents_batch = chartevents[chartevents['stay_id'].isin(stay_ids)]
    labevents_batch = labevents[(labevents['subject_id'].isin(subject_ids)) & (labevents['hadm_id'].isin(hadm_ids))]
    print(f"[LOG] Filtered chartevents_batch: {chartevents_batch.shape}, labevents_batch: {labevents_batch.shape} in {time.time() - t0:.2f}s")
    t0 = time.time()
    for idx, row in cohort_batch.iterrows():
        if idx - batch_start < 5:
            print(f"[LOG] process_patient START idx={idx}, subject_id={row['subject_id']}, stay_id={row['stay_id']}")
        ts, label = process_patient(row, chartevents_batch, labevents_batch)
        all_series.append(ts)
        all_labels.append(label)
        if idx - batch_start < 5:
            print(f"[LOG] process_patient END idx={idx}, subject_id={row['subject_id']}, stay_id={row['stay_id']}")
    print(f"[LOG] Finished batch {batch_start} to {batch_end} in {time.time() - t0:.2f}s")

print(f"[LOG] Finished all batches in {time.time() - t0_total:.2f}s")

X = np.stack(all_series)
y = np.array(all_labels)
np.save(os.path.join(OUTPUT_PATH, "X_mimiciv_parallel.npy"), X)
np.save(os.path.join(OUTPUT_PATH, "y_mimiciv_parallel.npy"), y)
print(f"Saved time series data: {X.shape}, labels: {y.shape}") 