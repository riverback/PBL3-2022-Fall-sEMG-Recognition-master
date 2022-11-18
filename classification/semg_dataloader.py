'''
用于生成输入sklearn机器学习模型的数据
用于生成输入神经网络的数据
'''

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

def folder_dataloader(folder):
    label = int(os.path.split(folder)[-1])
    csvfiles = os.listdir(folder)
    datas = []
    labels = []
    for csvfile in csvfiles:
        data = pd.read_csv(os.path.join(folder, csvfile), header=None).iloc[:, 2:]
        data = data.values
        datas.append(data.reshape(-1).tolist())
        labels.append(label)

    
    return datas, labels


def dataloader_ml(dataroot):
    folders = os.listdir(dataroot)
    datas = []
    labels = []
    for folder in folders:

        folder_data = folder_dataloader(os.path.join(dataroot, folder))
        datas += folder_data[0]
        labels += folder_data[1]
    
    return datas, labels


class sEMG_Dataset(Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.dataroot = dataroot
        self.datas, self.labels = self._prepare_data()
        
    def __getitem__(self, index):
        data = torch.tensor(self.datas[index], dtype=torch.float).unsqueeze(dim=0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        return (data, label)
        
    def __len__(self):
        return len(self.datas)
        
    def _folder_dataloader(self, folder):
        label = int(os.path.split(folder)[-1])
        csvfiles = os.listdir(folder)
        datas = []
        labels = []
        for csvfile in csvfiles:
            data = pd.read_csv(os.path.join(folder, csvfile), header=None).iloc[:, 2:]
            data = data.values
            N_channels = data.shape[-1]
            data = data.reshape(N_channels, -1)
            datas.append(data)
            labels.append(label)
        
        return (datas, labels)
    
    def _prepare_data(self):
        folders = os.listdir(self.dataroot)
        datas = []
        labels = []
        for folder in folders:
            folder_data = self._folder_dataloader(os.path.join(self.dataroot, folder))
            datas += folder_data[0]
            labels += folder_data[1]
        
        return datas, labels
    
def get_loader(dataroot, batch_size, mode='train', num_workers=0):
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    
    loader = DataLoader(
        dataset=sEMG_Dataset(dataroot=dataroot),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return loader


__all__ = [dataloader_ml, sEMG_Dataset, get_loader]

if __name__ == '__main__':
    mydataset = sEMG_Dataset(r'processed_data\slice_data')
    
    print('test')