import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# The dataset that reads image patches and their labels 
class patchDataset(Dataset):
    def __init__(self, data_dir, transform=None, labels_path='/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/labels.csv'):
        self.data = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        self.labels = pd.read_csv(labels_path)
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
                torch.flip(data, [-1])
            if rand[1]>0.5:
                torch.flip(data, [-2])
            if rand[2]>0.5:
                torch.flip(data, [-3])
            if rand[3]>0.5:
                torch.transpose(data, -1, -2)
            if rand[4]>0.5:
                torch.transpose(data, -1, -3)
            if rand[5]>0.5:
                torch.transpose(data, -2, -3)

        data = data[None]
        return data, label


# The dataset that reads feature vectors and their labels
class featureDataset(Dataset):
    def __init__(self, path='/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/features.csv'):
        df = pd.read_csv(path)
        self.labels = df["label"]
        self.prediction = df["prediction"]
        self.data = df["feature_vector"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the feature vector and matching label
        return np.array(json.loads(self.data[idx])), self.labels[idx]
