# data_processor.py
import pandas as pd
import numpy as np
from scipy.stats import scoreatpercentile

def affine_transform(time_series, alpha, beta):
    min_val = np.percentile(time_series, 0.1)
    max_val = np.percentile(time_series, 99)
    b = min_val - beta * (max_val - min_val)
    shifted_series = time_series - b
    a = scoreatpercentile(shifted_series, alpha)
    if np.all(np.abs(shifted_series) < 1e-3):
        a = min(a, 0.01)
    transformed_series = shifted_series / a
    transformed_series[time_series == 0] = 0
    return transformed_series

def load_and_preprocess_data(file_path, args):
    """
    Loads a single CSV file and applies all necessary preprocessing steps.
    """
    all_datas = pd.read_csv(file_path)
    all_datas.fillna(0, inplace=True)

    if args.delete_zero:
        all_datas = all_datas[all_datas['value'] != 0].reset_index(drop=True)
    
    if args.interpolate_zero:
        all_datas['value'] = all_datas['value'].replace(0, np.nan).interpolate(method='linear')
        
    all_datas.rename(columns={'value': 'raw_value'}, inplace=True)
    
    if not args.no_affine_transform:
        print("Applying affine transformation.")
        transform_series = affine_transform(all_datas['raw_value'].values, args.trans_alpha, args.trans_beta)
        all_datas['value'] = (transform_series * 1000).astype(int)
    else:
        print("Affine transformation is disabled. Applying simple min-max scaling.")
        min_val = all_datas['raw_value'].min()
        max_val = all_datas['raw_value'].max()
        if max_val - min_val > 0:
            all_datas['value'] = 1000 * (all_datas['raw_value'] - min_val) / (max_val - min_val)
        else:
            all_datas['value'] = 0
        all_datas['value'] = all_datas['value'].astype(int)
        
    return all_datas

def find_anomalies(data, pad_len=20, max_len=200):
    anomaly_sequences = []
    anomaly_str_sequences = []
    anomaly_labels = []
    start_index = None
    
    for index, row in data.iterrows():
        if row['label'] == 1:
            if start_index is None:
                start_index = index
        elif start_index is not None:
            if index - start_index > 1:
                pad_start = max(0, start_index - pad_len)
                pad_end = min(len(data), index + pad_len)
                
                sequence = data.iloc[pad_start:pad_end]
                if len(sequence) > max_len > 0:
                    sequence = sequence.iloc[:max_len]
                
                anomaly_sequences.append(sequence['value'].tolist())
                
                str_sequence = ",".join([f"*{val}*" if start_index <= i < index else str(val) for i, val in sequence['value'].items()])
                anomaly_str_sequences.append(str_sequence)
                
                labels = (sequence.index >= start_index) & (sequence.index < index)
                anomaly_labels.append(labels.astype(int).tolist())
                
            start_index = None
            
    return anomaly_sequences, anomaly_str_sequences, anomaly_labels

def find_zero_sequences(data, min_len=100, max_len=800, overlap=0):
    zero_sequences = []
    start_index = None

    for index, row in data.iterrows():
        if row['label'] == 0:
            if start_index is None:
                start_index = index
        else:
            if start_index is not None and index - start_index >= min_len:
                zero_sequences.append(data.iloc[start_index:index][['value']])
            start_index = None
            
    if start_index is not None and len(data) - start_index >= min_len:
        zero_sequences.append(data.iloc[start_index:][['value']])

    processed_sequences = []
    for seq_df in zero_sequences:
        if len(seq_df) > max_len > 0:
            for i in range(0, len(seq_df) - max_len + 1, max_len - overlap):
                processed_sequences.append(seq_df.iloc[i:i + max_len]['value'].tolist())
        else:
            processed_sequences.append(seq_df['value'].tolist())
            
    return processed_sequences
