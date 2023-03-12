import sys
sys.path.append('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/')

import tqdm
import os
import csv
import random
import torch


import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from torchvision.transforms import Compose, ToTensor, Normalize


# Import created classes
from swin_classifier.code import utils, dataset
from swin_classifier.model import MLPhead, SwinEncoder, CombinedModel

validation = utils.validation
MLPhead = MLPhead.MLPhead
SwinEncoder = SwinEncoder.SwinEncoder
CombinedModel = CombinedModel.CombinedModel
patchDataset = dataset.patchDataset
featureDataset = dataset.patchDataset


root_dir = "/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier"
os.chdir(root_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = None

# Create dataset
dataset = featureDataset('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/features', transforms)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

mlp_head = MLPhead(20736).to(device)

optimizer = torch.optim.Adam(mlp_head.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()

# Set training parameters
epochs = 20
random.seed(128)
mlp_head.train()
train_loss = []
val_loss = []
val_acc = []
best_acc = 0
total_loss = 0
for t in range(epochs):
    print(t)
    for i, (features, labels) in enumerate(tqdm(train_loader)):
        features, labels = features.to(device), labels.to(device)
        pred = mlp_head(features)
        loss = loss_fn(pred, torch.tensor(labels.unsqueeze(1), dtype=torch.float))
        total_loss = total_loss + loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track the training loss
        if i % 50 ==0:
            print("Loss: " + str(total_loss / 50))
            train_loss.append(total_loss / 50)
            total_loss = 0
            print(pred)
    # Each epoch, save the training loss to a CSV file        
    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/mlp_train.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in train_loss:
            writer.writerow([l])
    valLoss, valAcc = validation(val_loader, mlp_head, loss_fn, device)
    val_loss.append(valLoss)
    val_acc.append(valAcc)

    # Save the best performing model
    if valAcc > best_acc:
        best_acc = valAcc
        dict_path = '/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/best_mlp.pth'
        torch.save(mlp_head.state_dict(), dict_path)
    
    # Save the validation loss and accuracy
    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/mlp_validation.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in range(len(val_loss)):
            writer.writerow([val_loss[l], val_acc[l]])











