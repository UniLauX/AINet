#!/bin/bash

cd ./projects/SRNet/
model_list=('Salinas-4_model' 'Salinas-8_model' 'Salinas-12_model')

cd ./data_preprocess/
python preprocess.py --dataset 'Indian'
cd ../training/

for model in ${model_list[@]}
do
   echo $model 
   python LWNet_t_Indian.py --dataset 'Indian' --model_name 'LWNet_3' --lr 0.01 --transfer_model $model --epochs 60 --model_save_interval 2
   
done



