import os
import yaml
import numpy as np
import pandas as pd

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def update_claim_columns_v2(df):
    new_columns = []
    for col in df.columns:
        if '_x.3' in col:
            new_columns.append(col.replace('_x.3', '_Time_6'))
        elif '_y.3' in col:
            new_columns.append(col.replace('_y.3', '_Time_7'))
        elif '_x.2' in col:
            new_columns.append(col.replace('_x.2', '_Time_4'))
        elif '_y.2' in col:
            new_columns.append(col.replace('_y.2', '_Time_5'))
        elif '_x.1' in col:
            new_columns.append(col.replace('_x.1', '_Time_2'))
        elif '_y.1' in col:
            new_columns.append(col.replace('_y.1', '_Time_3'))
        elif '_x' in col:
            new_columns.append(col.replace('_x', '_Time_0'))
        elif '_y' in col:
            new_columns.append(col.replace('_y', '_Time_1'))
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df

def preprocess_data(config):
    raw = config['data']['raw_path']
    processed = config['data']['processed_path']
    month_count = config['data']['month_count']
    num_period = month_count // 3

    # Load raw datasets
    longitudinal_comorbidity_df = pd.read_csv(os.path.join(raw, f'longitudinal_comorbidity_df_2009_{month_count}.csv'))
    lab_test_longitudinal_df = pd.read_csv(os.path.join(raw, f'lab_test_longitudinal_df_2009_{month_count}.csv')).replace(0, np.nan)
    agr_longitudinal_df = pd.read_csv(os.path.join(raw, f'agr_longitudinal_df_{month_count}.csv'))
    stage_df = pd.read_csv(os.path.join(raw, 'stage_merged_df_matched.csv'))
    final_df = pd.read_csv(os.path.join(raw, f'final_df_{month_count}.csv'))
    agg_data = pd.read_csv(os.path.join(raw, f'Agg_data_{month_count}.csv'))

    # Drop specific columns
    longitudinal_comorbidity_df.drop(
        columns=[f'Transplant_Time_{i}' for i in range(num_period)] +
                [f'Dialysis_Time_{i}' for i in range(num_period)],
        inplace=True
    )

    lab_test_longitudinal_df.drop(
        columns=[f'Bicarbonate_Time_{i}' for i in range(num_period)] +
                [f'Urine_Albumin_Time_{i}' for i in range(num_period)],
        inplace=True
    )

    # Filter claim data
    claim_df = final_df[final_df.TMA_Acct.isin(agg_data.TMA_Acct)]
    claim_df = update_claim_columns_v2(claim_df)

    for s_old, s_new in [('S4_in_2years', 'S4'), ('S5_in_2years', 'S5')]:
        for i in range(num_period):
            claim_df.rename(columns={f'{s_old}_Time_{i}': f'{s_new}_Time_{i}'}, inplace=True)

    # Extract comorbidity and expenditure data from claims
    base_exp_columns = ['n_claims_DR', 'n_claims_I', 'n_claims_O', 'n_claims_P', 'net_exp_DR', 'net_exp_I', 'net_exp_O', 'net_exp_P']
    base_cor_columns = ['Diabetes', 'Hyptsn', 'CVD', 'Anemia', 'MA', 'Prot', 'Sec_Hyp', 'Phos', 'Atherosclerosis', 'CHF', 'Stroke', 'CD', 'MI', 'FE', 'MD', 'ND', 'S4', 'S5']

    exp_columns_to_keep = ['TMA_Acct'] + [f'{col}_Time_{i}' for col in base_exp_columns for i in range(num_period)]
    cor_columns_to_keep = ['TMA_Acct'] + [f'{col}_Time_{i}' for col in base_cor_columns for i in range(num_period)]

    claim_df_exp = claim_df[exp_columns_to_keep]
    claim_df_cor = claim_df[cor_columns_to_keep]
    claim_df_cor.columns = claim_df_cor.columns.str.replace('Hyptsn', 'Htn').str.replace('CVD', 'Cvd').str.replace('Atherosclerosis', 'Athsc').str.replace('Sec_Hyp', 'SH')

    # Remove patients with ESRD within observation period
    esrd_cols = [col for col in longitudinal_comorbidity_df.columns if 'ESRD' in col]
    longitudinal_comorbidity_df['ESRD'] = longitudinal_comorbidity_df[esrd_cols].max(axis=1)
    longitudinal_comorbidity_df = longitudinal_comorbidity_df[longitudinal_comorbidity_df.ESRD != 1].drop(columns=esrd_cols + ['ESRD'])

    # Merge comorbidity data
    merged_df = pd.merge(longitudinal_comorbidity_df, claim_df_cor, on='TMA_Acct', suffixes=('_long', '_claim'))
    conditions = [col for col in claim_df_cor.columns if col != 'TMA_Acct']

    cor_df = merged_df[['TMA_Acct']].copy()
    for condition in conditions:
        cor_df[condition] = merged_df[f"{condition}_long"].fillna(0).astype(int) | merged_df[f"{condition}_claim"].fillna(0).astype(int)

    # Merge all datasets
    data = cor_df \
        .merge(lab_test_longitudinal_df, on='TMA_Acct') \
        .merge(agr_longitudinal_df, on='TMA_Acct') \
        .merge(claim_df_exp, on='TMA_Acct') \
        .merge(stage_df[['TMA_Acct', 'ESRD']], on='TMA_Acct', how='left') \
        .merge(claim_df[['TMA_Acct', 'ESRD']], on='TMA_Acct', how='left', suffixes=('_x', '_y'))

    data['ESRD'] = ((data['ESRD_x'] == 1) | (data['ESRD_y'] == 1)).astype(int)
    data.drop(['ESRD_x', 'ESRD_y'], axis=1, inplace=True)

    # Sort columns by time
    time_suffixes = sorted({col.split('_')[-1] for col in data.columns if 'Time' in col})
    sorted_columns = ['TMA_Acct', 'ESRD'] + [col for suffix in time_suffixes for col in data.columns if col.endswith(suffix)]
    data = data[sorted_columns]

    # Save processed data
    os.makedirs(os.path.dirname(processed), exist_ok=True)
    data.to_csv(processed, index=False)
    
    

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(script_dir, '../config/config.yaml'))
    preprocess_data(config)
