#!/bin/bash

START_TIME=$(date +%s)

FINETUNE_DATA_NAME=BBBP
SPLIT_TYPES=("random" "scaffold" "Perimeter")
SEEDS=("2024" "2034" "2044" "2054" "2064")

for SPLIT_TYPE in "${SPLIT_TYPES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    LOG_FILE="logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log"
    
    echo "========================================"
    echo "Starting ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}"
    echo "Logging to: ${LOG_FILE}"
    echo "Current time: $(date)"
    
    # 执行任务（前台运行）
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune-time.py \
      --config ./config/ADMET_class.yaml \
      --split_type "${SPLIT_TYPE}_${SEED}" \
      --data_name "${FINETUNE_DATA_NAME}" > "${LOG_FILE}" 2>&1
    
    # 检查退出状态
    if [ $? -ne 0 ]; then
      echo "ERROR: Failed on ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}"
      echo "Last 10 lines of log:"
      tail -n 10 "${LOG_FILE}"
    else
      echo "Completed ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}"
    fi
    
    echo "Time elapsed so far: $(( $(date +%s) - START_TIME )) seconds"
    echo "========================================"
    echo
  done
done

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# 格式化时间显示
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo "========================================"
echo "All tasks completed!"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Average per task: $((ELAPSED_TIME / (${#SPLIT_TYPES[@]} * ${#SEEDS[@]}))) seconds"
echo "========================================"
