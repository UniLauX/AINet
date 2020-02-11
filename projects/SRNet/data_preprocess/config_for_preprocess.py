samples_dir = '../../../data/HSI_SRNet/samples/'
source_dir = '../../../data/HSI_SRNet/data_h5/'
dataset = 'KSC'

config = {
    'dataset_source_dir': source_dir + '{}.h5'.format(dataset),
    'samples_save_dir': samples_dir + '{}/'.format(dataset),
    'data_list_dir': './data_list/{}.txt'.format(dataset),
    'window_size': 27,
    'train_split_dir': './data_list/{}_split.txt'.format(dataset)
    }
