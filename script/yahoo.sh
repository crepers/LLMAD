#!/bin/bash

# Load environment variables from .env file
set -o allexport
source .env
set +o allexport

TRANS_BETA=0
TRANS_ALPHA=95
TEST_RATIO=0.5
WINDOW_SIZE=400
RETRIEVE_POSITIVE_NUM=2
RETRIEVE_NEGATIVE_NUM=1
RETRIEVE_DATABASE_RATIO=1
DELETE_ZERO=1
DATA_ROOT_DIR="data"
SAVE_DIR="result"

PROMPT_MODES=(3 4)
SUB_COMPANIES=("real" "s" "A3" "A4")

for i in 0 1; do
  for SUB_COMPANY in "${SUB_COMPANIES[@]}"; do
    if [ "$SUB_COMPANY" == "real" ]; then
      PROMPT_MODE=3
    else
      PROMPT_MODE=4
    fi
    TIMESTAMP=$(date +%Y%m%d%H%M)
    MODEL_SUFFIX=${MODEL_ENGINE%%-*}
    RUN_NAME="Yahoo_50_prompt_${PROMPT_MODE}_win_${WINDOW_SIZE}_beta${TRANS_BETA}alpha${TRANS_ALPHA}_p2n1_rate10_crossp_${MODEL_SUFFIX}_${TIMESTAMP}"
    python src/main.py \
      --trans_beta $TRANS_BETA \
      --trans_alpha $TRANS_ALPHA \
      --test_ratio $TEST_RATIO \
      --window_size $WINDOW_SIZE \
      --retrieve_positive_num $RETRIEVE_POSITIVE_NUM \
      --retrieve_database_ratio $RETRIEVE_DATABASE_RATIO \
      --retrieve_negative_num $RETRIEVE_NEGATIVE_NUM \
      --prompt_mode $PROMPT_MODE \
      --run_name $RUN_NAME \
      --infer_data_path "${DATA_ROOT_DIR}/Yahoo" \
      --retreive_data_path "${DATA_ROOT_DIR}/Yahoo" \
      --result_save_dir $SAVE_DIR \
      --sub_company $SUB_COMPANY \
      --delete_zero $DELETE_ZERO
  done
done

SOURCE_DIR1="Yahoo_50_prompt_${PROMPT_MODES[0]}_win_${WINDOW_SIZE}_beta${TRANS_BETA}alpha${TRANS_ALPHA}_p2n1_rate10_crossp_gpt_o"
SOURCE_DIR2="${SAVE_DIR}/Yahoo_50_prompt_${PROMPT_MODES[1]}_win_${WINDOW_SIZE}_beta${TRANS_BETA}alpha${TRANS_ALPHA}_p2n1_rate10_crossp_gpt_o"
TARGET_DIR="${SAVE_DIR}/Yahoo_50_win_${WINDOW_SIZE[0]}_beta${TRANS_BETA}alpha${TRANS_ALPHA}_p2n1_rate10_crossp_gpt_o"

mkdir -p "${TARGET_DIR}"

cp -rf ${SOURCE_DIR1}/* ${TARGET_DIR}
cp -rf ${SOURCE_DIR2}/* ${TARGET_DIR}

python Eval/Eval_yahoo.py --path "${TARGET_DIR}"
python Eval/Eval_yahoo2.py --path "${TARGET_DIR}"
