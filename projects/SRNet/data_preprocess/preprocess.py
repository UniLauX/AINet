import argparse
from functions_for_samples_extraction import samples_extraction, samples_division

# Training settings
samples_dir = '../../../data/HSI_SRNet/samples/'
source_dir = '../../../data/HSI_SRNet/data_h5/'

parser = argparse.ArgumentParser(description='HSI classification')
parser.add_argument('--dataset', type=str, default='KSC')
parser.add_argument('--window_size', type=int, default=27)
args = parser.parse_args()

dataset_source_dir = source_dir + '{}.h5'.format(args.dataset)
samples_save_dir = samples_dir + '{}/'.format(args.dataset)
data_list_dir = './data_list/{}.txt'.format(args.dataset)
window_size = args.window_size
train_split_dir = './data_list/{}_split_50.txt'.format(args.dataset)

# samples_extraction(dataset_source_dir, samples_save_dir, data_list_dir, window_size)
samples_division(data_list_dir, train_split_dir)


