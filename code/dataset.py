import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# The dataset that reads image patches and their labels 
class patchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        self.labels = pd.read_csv('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/labels.csv')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Read the image and find the appropriate label
        image = nib.load(self.data[idx])
        data = image.get_fdata()
        data = np.transpose(data, (2, 0, 1))
        no_ext = self.data[idx].split(".nii", 1)[0]
        label_id = int(no_ext.split("/patches/",1)[1])
        label = self.labels.iloc[label_id][1]


        # Apply any transformations to the image and label
        # Transform all possible rotations and flips
        if self.transform is not None:
            data = self.transform(data)
            rand = np.random.rand(6)
            if rand[0]>0.5:
                torch.flip(data, [0])
            if rand[1]>0.5:
                torch.flip(data, [1])
            if rand[2]>0.5:
                torch.flip(data, [2])
            if rand[3]>0.5:
                torch.transpose(data, 0, 1)
            if rand[4]>0.5:
                torch.transpose(data, 0, 2)
            if rand[5]>0.5:
                torch.transpose(data, 1, 2)
            

        return data, label

# The dataset that reads feature vectors and their labels
class featureDataset(Dataset):
    def __init__(self, data_dir, transform=None, path='/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/feature_labels.csv'):
        self.data = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.labels = pd.read_csv(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Read the feature vector and find the matching label
        tensor = torch.load(self.data[idx])
        
        no_ext = self.data[idx].split(".pt", 1)[0]
        label_id = int(no_ext.split("/features/",1)[1])
        label = self.labels.iloc[label_id][1]

        
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

