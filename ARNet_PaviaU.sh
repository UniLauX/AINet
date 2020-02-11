#!/bin/bash

cd ./projects/3D-ARNet/data_preprocess/
python preprocess.py --dataset 'PaviaU' --window_size 27

cd ../training/

python ARNet_main.py --dataset 'PaviaU' --model_name "ARNet_2" --lr 0.01 --epochs 60 --model_save_interval 2




