# ! /bin/bash

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=1 \
    python main.py \
    --config cfgs/gft/finetune_scan_objonly.yaml \
    --ckpts pretrained/Point-MAE/pretrained.pth \
    --finetune_model \
    --exp_name point_mae \
    --seed $i
done