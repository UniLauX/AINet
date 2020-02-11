#!/bin/bash

cd ./projects/3D-ARNet/data_preprocess/
# python preprocess.py --dataset 'KSC' --window_size 27

cd ../training/
python ARNet_main_t.py --dataset 'KSC' --model_name "ARNet_2" --devices '1' --transfer_model 'Salinas' --lr 0.01 --epochs 60 --model_save_interval 2




