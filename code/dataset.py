import os
import csv
import torch
import numpy as np
import pandas as pd
import nibabel as nib

from torch.utils.data import Dataset

class patchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        self.labels = pd.read_csv('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/labels.csv')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # read the image and label using the NiftiReader
        image = nib.load(self.data[idx])
        data = image.get_fdata()
        data = np.transpose(data, (2, 0, 1))
        no_ext = self.data[idx].split(".nii", 1)[0]
        label_id = int(no_ext.split("/patches/",1)[1])
        label = self.labels.iloc[label_id][1]


        # apply any transformations to the image and label
        if self.transform is not None:
            data = self.transform(data)

        return data, label

class featureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.labels = pd.read_csv('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/labels.csv')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # read the image and label using the NiftiReader
        tensor = torch.load(self.data[idx])
        
        no_ext = self.data[idx].split(".pt", 1)[0]
        label_id = int(no_ext.split("/features/",1)[1])
        label = self.labels.iloc[label_id][1]

        # apply any transformations to the image and label
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

