# ! /bin/bash

# Segmentation finetuning with Point-MAE
cd segmentation
for i in {0..9}
do 
    python main.py --cfg cfgs/gft.yaml \
     seed=$i
done

# Segmentation finetuning with Point-BERT
for i in {0..9}
do 
    python main.py --cfg cfgs/gft.yaml \
     ckpts=../pretrained/Point-BERT/pretrained.pth \
     model.encoder_dims=256 \
     seed=$i
done

