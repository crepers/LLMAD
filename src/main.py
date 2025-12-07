# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import sys
import pandas as pd
import json
import time
import numpy as np
from tqdm import tqdm

# --- Project Root Setup ---
# To ensure all paths are relative to the project root, not the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)
# --- End Setup ---

from llm_handler import get_llm_response
from data_processor import load_and_preprocess_data, find_anomalies, find_zero_sequences
from retriever import find_most_similar_series_fast
from prompt_template import PromptTemplate


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--infer_data_path', type=str, default="data/AIOPS_hard",required=False, help="Path to the inference data.")
    arg_parser.add_argument('--retreive_data_path', type=str, default="data/AIOPS",required=False, help="Path to the data to be retrieved.")
    arg_parser.add_argument('--sub_company', type=str, default="real",required=False, help="Subsidiary company.")
    arg_parser.add_argument('--window_size', type=int, default=400, required=False, help="Window size for the time series.")
    arg_parser.add_argument('--test_ratio', type=float, default=0.05, required=False, help="Ratio of the data to be used for testing.")
    arg_parser.add_argument('--trans_alpha', type=int, default=95, required=False, help="Alpha parameter for transformation.")
    arg_parser.add_argument('--trans_beta', type=float, default=0, required=False, help="Beta parameter for transformation.")
    arg_parser.add_argument('--retreive_ratio', type=float, default=0.50, required=False, help="Ratio of data to be retrieved.")
    arg_parser.add_argument('--prompt_mode', type=int, default=13, required=False, help="Mode for the prompt.")
    arg_parser.add_argument('--result_save_dir', type=str, default='LLM_AD/GPT_AD_exp', required=False, help="Directory to save the results.")
    arg_parser.add_argument('--run_name', type=str, default='KPI_hard_5_prompt_13_win_400_beta0alpha95_p1n1_retall_0114_test', required=False, help="Name of the run.")
    arg_parser.add_argument('--retrieve_positive_num', type=int, default=1, required=False, help="Number of positive instances to retrieve.")
    arg_parser.add_argument('--use_positive', action='store_false', default=True, required=False, help="Whether to use positive instances.")
    arg_parser.add_argument('--retrieve_negative_num', type=int, default=1, required=False, help="Number of negative instances to retrieve.")
    arg_parser.add_argument('--use_negative', action='store_false', default=True, required=False, help="Whether to use negative instances.")
    arg_parser.add_argument('--retrieve_negative_len', type=int, default=800, required=False, help="Length of negative instances to retrieve.")
    arg_parser.add_argument('--retrieve_negative_overlap', type=int, default=0, required=False, help="Overlap of negative instances to retrieve.")
    arg_parser.add_argument('--retrieve_database_ratio', type=float, default=0.1, required=False, help="Ratio of database to retrieve.")
    arg_parser.add_argument('--run_only_anomaly', type=bool, default=False, required=False, help="Whether to run only on anomalies.")
    arg_parser.add_argument('--overlap', type=bool, default=False, required=False, help="Whether to use overlap.")
    arg_parser.add_argument('--cross_retrieve', type=bool, default=True, required=False, help="Whether to use cross instance retrieval.")
    arg_parser.add_argument('--with_value', type=bool, default=False, required=False, help="Whether to predict value while testing.")
    arg_parser.add_argument('--dist_div_len', type=bool, default=False, required=False, help="Whether to divide the distance by length.")
    arg_parser.add_argument('--delete_zero', type=bool, default=False, required=False, help="Whether to delete zeros values.")
    arg_parser.add_argument('--interpolate_zero', type=bool, default=False, required=False, help="Whether to interpolate zeros values.")
    arg_parser.add_argument('--no_affine_transform', action='store_true', help="Disable the affine transformation.")
    arg_parser.add_argument('--cost_analysis', type=bool, default=False, required=False, help="Whether to perform cost analysis.")
    arg_parser.add_argument('--value_col', type=str, default="value", required=False, help="Name of the value column.")
    arg_parser.add_argument('--label_col', type=str, default="label", required=False, help="Name of the label column.")
    arg_parser.add_argument('--prompt_extra_cols', nargs='*', default=[], required=False, help="List of extra columns to include in the prompt.")
    arg_parser.add_argument('--data_description', type=str, default="", required=False, help="Description of the data for the prompt.")
    arg_parser.add_argument('--max_workers', type=int, default=5, required=False, help="Number of workers for parallel processing.")
    arg_parser.add_argument('--max_windows', type=int, default=0, required=False, help="Maximum number of windows to process. 0 means all.")

    return arg_parser.parse_args()

def prepare_retrieval_database(args):
    ad_list, ad_str_list, ad_label_list = [], [], []
    
    # Use absolute path
    retrieval_path_abs = os.path.join(PROJECT_ROOT, args.retreive_data_path)
    retrieve_files = os.listdir(retrieval_path_abs)
    for file in tqdm(retrieve_files, desc="Building retrieval database"):
        if not file.startswith(args.sub_company) and args.sub_company != 'all':
            continue
        
        file_path = os.path.join(retrieval_path_abs, file)
        processed_data = load_and_preprocess_data(file_path, args)
        
        train_data = processed_data[:int(args.retreive_ratio * len(processed_data))]
        ad_list_, ad_str_list_, ad_label_list_ = find_anomalies(train_data, pad_len=20, max_len=200)
        
        ad_list.extend([np.squeeze(np.asarray(s)) for s in ad_list_])
        ad_str_list.extend(ad_str_list_)
        ad_label_list.extend(ad_label_list_)

    # Shuffle and sample
    shuffle_indices = np.random.permutation(len(ad_list))
    num_samples = int(len(ad_list) * args.retrieve_database_ratio)
    
    ad_list = np.asarray(ad_list, dtype=object)[shuffle_indices][:num_samples]
    ad_str_list = np.asarray(ad_str_list, dtype=object)[shuffle_indices][:num_samples]
    ad_label_list = np.asarray(ad_label_list, dtype=object)[shuffle_indices][:num_samples]
    
    print(f'Total anomaly examples in database: {len(ad_list)}')
    return ad_list, ad_str_list, ad_label_list

def run_inference_on_window(data_window, T_list, ad_list, ad_str_list, ad_label_list, prompt_template, args):
    """
    Processes a single window of data: retrieves examples, calls LLM, and returns results.
    """
    X = data_window.value.tolist()
    
    # Retrieve similar series
    normal_series, scores, _ = find_most_similar_series_fast(X, T_list, top_k=args.retrieve_negative_num, dist_div_len=args.dist_div_len)
    anomaly_series, ad_scores, ad_indices = find_most_similar_series_fast(X, ad_list, top_k=args.retrieve_positive_num, dist_div_len=args.dist_div_len)

    # Format data for prompt
    normal_data_str = "\n".join([f"series_{j+1}:{','.join(map(str, np.array(s).astype(int)))}" for j, s in enumerate(normal_series)])
    anomaly_data_str = [ad_str_list[j] for j in ad_indices]
    cur_data_str = "\n".join([f"{j+1} {val}" for j, val in enumerate(data_window.value)])
    
    if args.prompt_extra_cols:
        # Include extra columns in the data string
        cur_data_lines = []
        for j, (idx, row) in enumerate(data_window.iterrows()):
            extra_info = " ".join([f"{col}={row[col]}" for col in args.prompt_extra_cols if col in row])
            cur_data_lines.append(f"{j+1} {row['value']} {extra_info}")
        cur_data_str = "\n".join(cur_data_lines)

    # Get prompt and response
    prompt_res = prompt_template.get_template(normal_data=normal_data_str, data=cur_data_str, data_len=len(data_window), anomaly_datas=anomaly_data_str, data_description=args.data_description)
    
    start_time = time.time()
    response, raw_response = get_llm_response(prompt_res, args)
    end_time = time.time()

    # Prepare results
    infer_result = None
    log_entry = None

    if response:
        is_anomaly = response.get('is_anomaly', False)
        anomalies = response.get('anomalies', [])
        
        predict_col = [0] * len(data_window)
        if is_anomaly and anomalies:
            refer_data = data_window.iloc[0]['idx']
            anomaly_indices = {index + refer_data for index in anomalies}
            for i, row_idx in enumerate(data_window['idx']):
                if row_idx in anomaly_indices:
                    predict_col[i] = 1
        
        infer_result = data_window.copy()
        infer_result['predict'] = predict_col
        infer_result['ad_len'] = len(anomalies) if is_anomaly else 0
        infer_result['alarm_level'] = response.get('alarm_level', 'no')
        infer_result['anomaly_type'] = response.get('anomaly_type', 'no')

        log_entry = {
            'ground_truth': json.dumps(data_window[data_window['label'] == 1].index.tolist()),
            'predict': json.dumps(anomalies),
            'data': json.dumps(X),
            'normal_data': json.dumps([s.tolist() if isinstance(s, np.ndarray) else s for s in normal_series]),
            'anomaly_data': json.dumps([s.tolist() for s in anomaly_series]),
            'anomaly_label': json.dumps([ad_label_list[i] for i in ad_indices]),
            'scores': json.dumps(scores),
            'ad_scores': json.dumps(ad_scores),
            'briefExplanation': response.get('briefExplanation', ''),
            'anomaly_type': response.get('anomaly_type', 'no'),
            'reason': response.get('reason', ''),
            'alarm_level': response.get('alarm_level', 'no'),
            'prompt_res': prompt_res,
            'raw_response': raw_response,
            'elapsed_time': end_time - start_time,
        }
    
    return infer_result, log_entry

def process_file(file, run_dir, ad_list, ad_str_list, ad_label_list, args):
    """
    Loads and processes a single data file for anomaly detection using parallel processing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    file_name = file.split('.')[0]
    base_dir = os.path.join(run_dir, file_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # Load and preprocess data
    file_path = os.path.join(PROJECT_ROOT, args.infer_data_path, file)
    all_datas = load_and_preprocess_data(file_path, args)
    print(" -> Data loaded and preprocessed.")
    
    infer_data = all_datas[int((1 - args.test_ratio) * len(all_datas)):].reset_index(drop=True)
    train_data = all_datas[:int(args.retreive_ratio * len(all_datas))]
    infer_data['idx'] = infer_data.index

    # Prepare local retrieval data if not using cross-retrieve
    print(" -> Preparing retrieval data (normal/anomaly examples)...")
    if not args.cross_retrieve:
        ad_list, ad_str_list, ad_label_list = find_anomalies(train_data, pad_len=20, max_len=0)
        ad_list = [np.squeeze(np.asarray(s)) for s in ad_list]

    T_list = find_zero_sequences(train_data, min_len=args.retrieve_negative_len, max_len=args.retrieve_negative_len, overlap=0)
    print(f" -> Found {len(T_list)} normal examples and {len(ad_list)} anomaly examples.")
    
    # Sliding window inference
    prompt_template = PromptTemplate(prompt_mode=args.prompt_mode)
    increment = args.window_size // 2 if args.overlap else args.window_size
    
    windows = []
    for i in range(0, len(infer_data), increment):
        data_window = infer_data[i : i + args.window_size]
        if not data_window.empty:
            windows.append(data_window)

    # Apply max_windows limit
    if args.max_windows > 0:
        print(f" -> Limiting to {args.max_windows} windows (out of {len(windows)}).")
        windows = windows[:args.max_windows]

    print(f" -> Processing {len(windows)} windows with {args.max_workers} workers...")
    
    result_path = os.path.join(base_dir, 'predict.csv')
    log_path = os.path.join(base_dir, 'log.csv')
    
    # Clear existing files
    if os.path.exists(result_path): os.remove(result_path)
    if os.path.exists(log_path): os.remove(log_path)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(run_inference_on_window, window, T_list, ad_list, ad_str_list, ad_label_list, prompt_template, args)
            for window in windows
        ]
        
        for future in tqdm(as_completed(futures), total=len(windows), desc="Inference Progress"):
            try:
                infer_result, log_entry = future.result()
                
                if infer_result is not None:
                    header = not os.path.exists(result_path)
                    infer_result.to_csv(result_path, mode='a', header=header, index=False)
                
                if log_entry is not None:
                    log_header = not os.path.exists(log_path)
                    pd.DataFrame([log_entry]).to_csv(log_path, mode='a', header=log_header, index=False)
                    
            except Exception as e:
                print(f"Error processing window: {e}")

def main(args):
    # Use absolute path
    run_dir_abs = os.path.join(PROJECT_ROOT, args.result_save_dir, args.run_name)
    os.makedirs(run_dir_abs, exist_ok=True)

    ad_list, ad_str_list, ad_label_list = [], [], []
    if args.cross_retrieve:
        ad_list, ad_str_list, ad_label_list = prepare_retrieval_database(args)
    
    infer_path_abs = os.path.join(PROJECT_ROOT, args.infer_data_path)
    all_files = os.listdir(infer_path_abs)
    files_to_process = [f for f in all_files if f.startswith(args.sub_company) or args.sub_company == 'all']
    total_files = len(files_to_process)
    print(f"Found {total_files} files to process.")

    for idx, file in enumerate(files_to_process):
        print(f"\n[{idx + 1}/{total_files}] Processing file: {file.split('.')[0]}")
        process_file(file, run_dir_abs, ad_list, ad_str_list, ad_label_list, args)

    print('All files processed.')

if __name__ == '__main__':
    args = parse_args()
    main(args)