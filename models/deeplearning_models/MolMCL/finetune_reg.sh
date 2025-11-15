############ Caco2
START_TIME=$(date +%s)

FINETUNE_DATA_NAME=Caco2
for SPLIT_TYPE in Perimeter scaffold random
do
  for SEED in 2024 2034 2044 2054 2064
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_reg.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &

    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Script execution time: ${ELAPSED_TIME} seconds"

############ HalfLife
START_TIME=$(date +%s)

FINETUNE_DATA_NAME=HalfLife
for SPLIT_TYPE in Perimeter scaffold random
do
  for SEED in 2024 2034 2044 2054 2064
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_reg.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &

    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Script execution time: ${ELAPSED_TIME} seconds"


############ VDss
START_TIME=$(date +%s)

FINETUNE_DATA_NAME=VDss
for SPLIT_TYPE in Perimeter scaffold random
do
  for SEED in 2024 2034 2044 2054 2064
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_reg.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &

    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Script execution time: ${ELAPSED_TIME} seconds"