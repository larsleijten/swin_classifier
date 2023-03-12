import sys
sys.path.append('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/')

import monai
import nibabel
import tqdm
import os
import shutil
import tempfile
import random
import csv

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from torchvision.transforms import Compose, ToTensor, Normalize

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch

from swin_classifier.code import utils, dataset
from swin_classifier.model import MLPhead, SwinEncoder, CombinedModel


validation = utils.validation
MLPhead = MLPhead.MLPhead
SwinEncoder = SwinEncoder.SwinEncoder
CombinedModel = CombinedModel.CombinedModel
patchDataset = dataset.patchDataset
featureDataset = dataset.featureDataset


print_config()

root_dir = "/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier"
os.chdir(root_dir)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = Compose([
    ToTensor(), 
    Normalize(mean=[0.5], std=[0.5])
])

dataset = patchDataset('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/patches', transforms)
# define the sizes of the training and validation sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

swin_encoder = SwinEncoder(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
).to(device)

swin_weights = torch.load("/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/model_swinvit.pt")
swin_encoder.load_from(weights=swin_weights)
print("Using pretrained self-supervied Swin UNETR backbone weights")



mlp_head = MLPhead(20736).to(device)

#mlp_head.load_state_dict(torch.load('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/best_mlp.pth'))

model = CombinedModel(swin_encoder, mlp_head)

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()

epochs = 50
random.seed(128)
model.train()
train_loss = []
val_loss = []
val_acc = []
best_acc = 0
total_loss = 0
for t in range(epochs):
    print("Epoch number: " + str(t))
    for i, (batch, labels) in enumerate(tqdm(train_loader)):
        batch, labels = batch.to(device, dtype=torch.half), labels.to(device)
        batch = batch[None, :].to(device=device, dtype = torch.half)
        with torch.cuda.amp.autocast():
            pred = model(batch)
        
        loss = loss_fn(pred, torch.tensor(labels, dtype=torch.half))
        total_loss = total_loss + loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 ==0:
            #print("Train loss: " + str(loss.item()))
            train_loss.append(total_loss / 50)
            total_loss = 0

    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/combined_train.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in train_loss:
            writer.writerow([l])
    model.eval()
    valLoss, valAcc = validation(val_loader, model, loss_fn, device)
    model.train()
    val_loss.append(valLoss)
    val_acc.append(valAcc)

    if valAcc > best_acc:
        best_acc = valAcc
        dict_path = '/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/best_combined_model.pth'
        torch.save(model.state_dict(), dict_path)

    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/combined_validation.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in range(len(val_loss)):
            writer.writerow([val_loss[l], val_acc[l]])

dict_path = '/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/last_combined_model.pth'
torch.save(model.state_dict(), dict_path)


