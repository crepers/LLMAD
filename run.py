# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import pandas as pd
import json
import time
import numpy as np
from tqdm import tqdm

from llm_handler import get_llm_response
from data_processor import load_and_preprocess_data, find_anomalies, find_zero_sequences
from retriever import find_most_similar_series_fast
from Prompt_template import PromptTemplate


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

    return arg_parser.parse_args()

def prepare_retrieval_database(args):
    ad_list, ad_str_list, ad_label_list = [], [], []
    
    retrieve_files = os.listdir(args.retreive_data_path)
    for file in tqdm(retrieve_files, desc="Building retrieval database"):
        if not file.startswith(args.sub_company) and args.sub_company != 'all':
            continue
        
        file_path = os.path.join(args.retreive_data_path, file)
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

def main():
    args = parse_args()
    print(args)

    # Setup directories
    run_dir = os.path.join(args.result_save_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Prepare retrieval database
    if args.cross_retrieve:
        ad_list, ad_str_list, ad_label_list = prepare_retrieval_database(args)
    
    # Process each file
    all_files = os.listdir(args.infer_data_path)
    files_to_process = [f for f in all_files if f.startswith(args.sub_company) or args.sub_company == 'all']
    total_files = len(files_to_process)
    print(f"Found {total_files} files to process.")

    for idx, file in enumerate(files_to_process):
        file_name = file.split('.')[0]
        print(f"\n[{idx + 1}/{total_files}] Processing file: {file_name}")

        base_dir = os.path.join(run_dir, file_name)
        os.makedirs(base_dir, exist_ok=True)
        
        # Load and preprocess data
        file_path = os.path.join(args.infer_data_path, file)
        all_datas = load_and_preprocess_data(file_path, args)
        print(" -> Data loaded and preprocessed.")
        
        infer_data = all_datas[int((1 - args.test_ratio) * len(all_datas)):].reset_index(drop=True)
        train_data = all_datas[:int(args.retreive_ratio * len(all_datas))]

        # Add index column to inference data
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
        num_windows = (len(infer_data) + increment - 1) // increment
        
        for i in range(0, len(infer_data), increment):
            window_num = (i // increment) + 1
            print(f"   -> Processing window {window_num} / {num_windows}...")
            data_window = infer_data[i : i + args.window_size]
            if data_window.empty: continue

            X = data_window.value.tolist()
            
            # Retrieve similar series
            normal_series, scores, _ = find_most_similar_series_fast(X, T_list, top_k=args.retrieve_negative_num, dist_div_len=args.dist_div_len)
            anomaly_series, ad_scores, ad_indices = find_most_similar_series_fast(X, ad_list, top_k=args.retrieve_positive_num, dist_div_len=args.dist_div_len)

            # Format data for prompt
            normal_data_str = "\n".join([f"series_{j+1}:{','.join(map(str, np.array(s).astype(int)))}" for j, s in enumerate(normal_series)])
            anomaly_data_str = [ad_str_list[j] for j in ad_indices]
            cur_data_str = "\n".join([f"{j+1} {val}" for j, val in enumerate(data_window.value)])
            
            # Get prompt and response
            prompt_res = prompt_template.get_template(normal_data=normal_data_str, data=cur_data_str, data_len=len(data_window), anomaly_datas=anomaly_data_str)
            
            start_time = time.time()
            print("      -> Calling LLM for anomaly detection...")
            response, raw_response = get_llm_response(prompt_res, args)
            end_time = time.time()
            print("      -> LLM response received. Saving results...")

            # --- Add Result Saving Logic Start ---
            result_path = os.path.join(base_dir, 'predict.csv')
            log_path = os.path.join(base_dir, 'log.csv')

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

                # Append to predict.csv
                header = not os.path.exists(result_path)
                infer_result.to_csv(result_path, mode='a', header=header, index=False)

                # Append to log.csv
                log_header = not os.path.exists(log_path)
                log_df = pd.DataFrame([{
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
                }])
                log_df.to_csv(log_path, mode='a', header=log_header, index=False)
            # --- Add Result Saving Logic End ---

    print('All files processed.')

if __name__ == '__main__':
    main()