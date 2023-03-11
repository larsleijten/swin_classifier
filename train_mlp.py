import monai
import nibabel
import tqdm
import os
import csv
import random
import torch
import torch.nn as nn


import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from torchvision.transforms import Compose, ToTensor, Normalize

def validation(val_loader, mlp_head, loss_fn, device):
    size = len(val_loader.dataset)
    num_batches = len(val_loader)
    mlp_head.eval()
    test_loss, correct = 0, 0
    # Do not keep track of gradients
    with torch.no_grad():
        # Loop over the batches in the dataloader
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            # Model predictions
            pred = mlp_head(X)

            # Make a binary prediction at the threshold of 0.5
            bin_pred = torch.round(pred).transpose(0,1)
            # Keep track of loss and accuracy
            test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
            correct += (bin_pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, (100*correct)

root_dir = "/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier"
os.chdir(root_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = None

dataset = featureDataset('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/features', transforms)
# define the sizes of the training and validation sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

mlp_head = MLPhead(20736).to(device)

optimizer = torch.optim.Adam(mlp_head.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()

epochs = 20
random.seed(128)
mlp_head.train()
train_loss = []
val_loss = []
val_acc = []
best_acc = 0
for t in range(epochs):
    print(t)
    for i, (features, labels) in enumerate(tqdm(train_loader)):
        features, labels = features.to(device), labels.to(device)
        pred = mlp_head(features)
        loss = loss_fn(pred, torch.tensor(labels.unsqueeze(1), dtype=torch.float))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 ==0:
            print("Loss: " + str(loss.item()))
            train_loss.append(loss.item())
    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/mlp_train.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in train_loss:
            writer.writerow([l])
    valLoss, valAcc = validation(val_loader, mlp_head, loss_fn, device)
    val_loss.append(valLoss)
    val_acc.append(valAcc)

    if valAcc > best_acc:
        best_acc = valAcc
        dict_path = '/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/best_mlp.pth'
        torch.save(mlp_head.state_dict(), dict_path)

    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/mlp_validation.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in range(len(val_loss)):
            writer.writerow([val_loss[l], val_acc[l]])











