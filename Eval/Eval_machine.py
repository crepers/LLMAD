import argparse
import os
import pandas as pd
import numpy as np
from Metric import calculate_f1, point_adjust

def compute_metric_machine(path):
    all_data = []
    
    # Check if path is a directory of results (contains subdirectories)
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if 'predict.csv' in files:
                file_path = os.path.join(root, 'predict.csv')
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not all_data:
        print("No predict.csv files found.")
        return

    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Check if label column exists, if not assume 0
    if 'label' not in combined_data.columns:
        combined_data['label'] = 0
        
    y_true = combined_data['label'].tolist()
    y_pred = combined_data['predict'].tolist()

    # Point Adjustment
    adjust_y_pred, adjust_y_true = point_adjust(y_pred, y_true)
    score, precision, recall = calculate_f1(adjust_y_true, adjust_y_pred)

    print(f"Aggregated Results over {len(all_data)} files:")
    print(f"F1 Score: {score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Save metrics
    metrics = {
        'F1': score,
        'Precision': precision,
        'Recall': recall
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(path, 'metrics_aggregated.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to the result directory")
    args = parser.parse_args()
    compute_metric_machine(args.path)
