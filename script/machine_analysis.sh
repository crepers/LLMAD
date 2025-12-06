#!/bin/bash

# Example usage for Machine data analysis
# Assumes data is in data/air and has columns: Datetime, param1, Machine, Stage

# Run inference
python src/main.py \
    --infer_data_path data/air \
    --retreive_data_path data/air \
    --sub_company sample \
    --window_size 100 \
    --prompt_mode 5 \
    --result_save_dir result/machine_test \
    --run_name run_1 \
    --value_col param1 \
    --label_col label \
    --prompt_extra_cols Machine Stage \
    --data_description "The data contains sensor readings from a machine." \
    --no_affine_transform \
    --retrieve_positive_num 0 \
    --retrieve_negative_num 0 \
    --cross_retrieve False \
    --test_ratio 1.0

# Evaluate
python Eval/Eval_machine.py --path result/machine_test/run_1
