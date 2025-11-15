############# BBBP
START_TIME=$(date +%s)

FINETUNE_DATA_NAME=BBBP
for SPLIT_TYPE in Perimeter scaffold random
do
  for SEED in 2024 2034 2044 2054 2064
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=1 python ./scripts/finetune.py \
      --config ./config/ADMET_class.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &

    sleep 1
  done
done

#wait


############ oral_bioavailability
FINETUNE_DATA_NAME=oral_bioavailability
for SPLIT_TYPE in Perimeter scaffold random
do
  for SEED in 2024 2034 2044 2054 2064
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_class.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &

    sleep 1
  done
done

# HLM_metabolic_stability
FINETUNE_DATA_NAME=HLM_metabolic_stability
for SPLIT_TYPE in Perimeter scaffold random
do
  for SEED in 2024 2034 2044 2054 2064
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_class.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &

    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Script execution time: ${ELAPSED_TIME} seconds"


############ hERG
START_TIME=$(date +%s)

FINETUNE_DATA_NAME=hERG
for SPLIT_TYPE in Perimeter scaffold random
do
  for SEED in 2024 2034 2044 2054 2064
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_class.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &

    sleep 1
  done
done

#wait


#END_TIME=$(date +%s)  # Capture end time
#ELAPSED_TIME=$((END_TIME - START_TIME))
#
#echo "Script execution time: ${ELAPSED_TIME} seconds"


 Mutagenicity
START_TIME=$(date +%s)

FINETUNE_DATA_NAME=Mutagenicity
for SPLIT_TYPE in Perimeter scaffold random
do
  for SEED in 2024 2034 2044 2054 2064
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=1 python ./scripts/finetune.py \
      --config ./config/ADMET_class.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &
    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Script execution time: ${ELAPSED_TIME} seconds"