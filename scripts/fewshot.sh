# ! /bin/bash

for WAY in 5 10
do
  for SHOT in 10 20
  do
    for FOLD in $(seq 0 9)
    do
	  CUDA_VISIBLE_DEVICES=0 \
      python main.py \
      --config cfgs/gft/fewshot.yaml \
      --finetune_model \
      --ckpts pretrained/Point-MAE/pretrained.pth \
      --exp_name point_mae \
      --way ${WAY} \
      --shot ${SHOT} \
      --fold ${FOLD}
    done
  done
done