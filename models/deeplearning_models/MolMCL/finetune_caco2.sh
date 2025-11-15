FINETUNE_DATA_NAME=Caco2
for SPLIT_TYPE in random scaffold Perimeter
do
  for SEED in 2034 2044 2054 2064 2024
  do
    echo ${SPLIT_TYPE}_${SEED}_${FINETUNE_DATA_NAME}
    CUDA_VISIBLE_DEVICES=0 python ./scripts/finetune.py \
      --config ./config/ADMET_reg.yaml \
      --split_type ${SPLIT_TYPE}_${SEED} \
      --data_name ${FINETUNE_DATA_NAME}
  done
done
