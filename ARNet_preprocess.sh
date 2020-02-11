#!/bin/bash

cd ./projects/3D-ARNet/data_preprocess/
python preprocess.py --dataset 'PaviaU' --window_size 27
python preprocess.py --dataset 'Indian' --window_size 27
python preprocess.py --dataset 'KSC' --window_size 27





