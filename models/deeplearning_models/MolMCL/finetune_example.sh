START_TIME=$(date +%s)

FINETUNE_DATA_NAME=BBBP  
for SPLIT_TYPE in Perimeter scaffold random 
do
  for SEED in 2024 2034 2044 2054 2064
  do
    # At this point, the `finetune.py` script will look for data `BBBP_${SPLIT_TYPE}_${SEED}.csv`,
    # as well as the specified configuration file `./config/ADMET_example.yaml`. 
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}

    CUDA_VISIBLE_DEVICES=1 python ./scripts/finetune.py \
      --config ./config/ADMET_example.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME} > logs/${FINETUNE_DATA_NAME}_${SPLIT_TYPE}_${SEED}.log 2>&1 &

    sleep 1
  done
done

wait

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Script execution time: ${ELAPSED_TIME} seconds"