import torch
import numpy as np
import torch.utils.data as data

class data_loader(data.Dataset):
    def __init__(self, list_dir, category_start=0, augmentation=False):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)
        self.au = augmentation
        self.category_start = category_start

    def __getitem__(self, index):
        sample_path = self.list_txt[index].split(' ')
        data_path = sample_path[0]
        label = sample_path[1][:-1]

        if not self.au:
            data = np.load(data_path)
        else:
            data = self.random_flip_lr(np.load(data_path))
            data = self.random_flip_tb(data)
            data = self.random_rot(data)

        label = int(label)-1
        return torch.from_numpy(data).float(), label+self.category_start

    def __len__(self):
        return self.length

    def random_flip_lr(self, data):
        if np.random.randint(0, 2):
            c, d, h, w = data.shape
            index = np.arange(w, 0, -1)-1
            return data[:,:,:, index]
        else:
            return data

    def random_flip_tb(self, data):
        if np.random.randint(0, 2):
            c, d, h, w = data.shape
            index = np.arange(h, 0, -1)-1
            return data[:,:,index,:]
        else:
            return data

    def random_rot(self, data):
        rot_k = np.random.randint(0, 4)
        return np.rot90(data, rot_k, (2, 3)).copy()