#!/usr/bin bash

for s in 0
do 
    python DeepAligned.py \
        --dataset vcb \
        --known_cls_ratio 0.75 \
        --cluster_num_factor 1 \
        --seed $s \
        --freeze_bert_parameters \
        --save_model
done
