# job#1:
START_TIME=$(date +%s)

SPLIT_TYPE=ac
for FINETUNE_DATA_NAME in CHEMBL2047_EC50 CHEMBL2034_Ki CHEMBL214_Ki CHEMBL235_EC50
  do
  for SEED in 2054 2064 2034 2044 2024
  do
    echo ${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=1 python ./scripts/finetune.py \
      --config ./config/ADMET_ac.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &
    sleep 1
  done
done


SPLIT_TYPE=ac
for FINETUNE_DATA_NAME in CHEMBL237_EC50 CHEMBL238_Ki CHEMBL3979_EC50 CHEMBL4005_Ki
  do
  for SEED in 2054 2064 2034 2044 2024
  do
    echo ${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_ac.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &
    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Script execution time: ${ELAPSED_TIME} seconds"


# job#2:
START_TIME=$(date +%s)

SPLIT_TYPE=ac
for FINETUNE_DATA_NAME in CHEMBL1862_Ki CHEMBL1871_Ki CHEMBL204_Ki CHEMBL2147_Ki
  do
  for SEED in 2054 2064 2034 2044 2024
  do
    echo ${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=1 python ./scripts/finetune.py \
      --config ./config/ADMET_ac.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &
    sleep 1
  done
done


SPLIT_TYPE=ac
for FINETUNE_DATA_NAME in CHEMBL218_EC50 CHEMBL219_Ki CHEMBL228_Ki CHEMBL231_Ki
  do
  for SEED in 2054 2064 2034 2044 2024
  do
    echo ${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_ac.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &
    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Script execution time: ${ELAPSED_TIME} seconds"


# job#3:
START_TIME=$(date +%s)

SPLIT_TYPE=ac
for FINETUNE_DATA_NAME in CHEMBL244_Ki CHEMBL262_Ki CHEMBL264_Ki CHEMBL2835_Ki CHEMBL287_Ki CHEMBL233_Ki CHEMBL234_Ki
  do
  for SEED in 2054 2064 2034 2044 2024
  do
    echo ${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=1 python ./scripts/finetune.py \
      --config ./config/ADMET_ac.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &
    sleep 1
  done
done

SPLIT_TYPE=ac
for FINETUNE_DATA_NAME in CHEMBL236_Ki CHEMBL237_Ki CHEMBL239_EC50
  do
  for SEED in 2054 2064 2034 2044 2024
  do
    echo ${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_ac.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &
    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Script execution time: ${ELAPSED_TIME} seconds"


# job#4:
START_TIME=$(date +%s)

SPLIT_TYPE=ac
for FINETUNE_DATA_NAME in CHEMBL2971_Ki CHEMBL4203_Ki CHEMBL4616_EC50 CHEMBL4792_Ki
  do
  for SEED in 2054 2064 2034 2044 2024
  do
    echo ${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=1 python ./scripts/finetune.py \
      --config ./config/ADMET_ac.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &
    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Script execution time: ${ELAPSED_TIME} seconds"