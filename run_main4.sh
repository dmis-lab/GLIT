#!/bin/bash

lr="5e-4"
wd="1e-5"
epoch=20

python main4.py \
    --model="GEX_PPI_GAT_cat4_MLP" \
    --attn_dim=0 \
    --loss_alpha=0 \
    --learning_rate=$lr \
    --weight_decay=$wd \
    --n_epochs=$epoch \
    --eval="True"
	
