import sys
from pathlib import Path

sys.path.append('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/')

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Import created classes
from swin_classifier.code.dataset import featureDataset
from swin_classifier.code.utils import validation
from swin_classifier.model.MLPhead import MLPhead


# Declare other validation function, due to different data shapes
def validation(val_loader, model, loss_fn, device):
    size = len(val_loader.dataset)
    num_batches = len(val_loader)
    model.eval()
    test_loss, correct, TP, TN, FP, FN, total = 0, 0, 0, 0, 0, 0, 0
    # Do not keep track of gradients
    with torch.no_grad():
        # Loop over the batches in the dataloader
        for X, y in val_loader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)

            # Model predictions
            with torch.cuda.amp.autocast():
                pred = model(X).to(device, dtype=torch.float)

            sigmoid = nn.Sigmoid()
            pred = sigmoid(pred)

            # Make a binary prediction at the threshold of 0.5
            bin_pred = torch.round(pred.unsqueeze(1))#.transpose(0,1)

            # Keep track of loss and accuracy
            test_loss += loss_fn(pred.squeeze().unsqueeze(1), y.unsqueeze(1).float()).item()
            correct += (bin_pred.squeeze() == y.squeeze()).type(torch.float).sum().item()
            total += len(y)
            #TP += (bin_pred == y and bin_pred == torch.tensor(1.0)).type(torch.float).sum().item()
            #TN += (bin_pred == y and bin_pred == torch.tensor(0.0)).type(torch.float).sum().item()
            #FP += (bin_pred != y and bin_pred == torch.tensor(1.0)).type(torch.float).sum().item()
            #FN += (bin_pred != y and bin_pred == torch.tensor(0.0)).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / total
    #sensitivity = TP / (TP + FN + 0.0001)
    #specificity = TN / (TN + FP + 0.0001)
    print(f"Validation Error: \n Accuracy: {(accuracy):>0.1%}, Avg loss: {test_loss:>8f}" )

    return test_loss, accuracy


root_dir = Path("/Users/joeranbosma/tmp/")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(64)
np.random.seed(64)
torch.manual_seed(64)


# Create dataset
dataset = featureDataset(root_dir / 'data/ct_images/features.csv')
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

mlp_head = MLPhead(20736).to(device)

optimizer = torch.optim.Adam(mlp_head.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss().to(device)

# Set training parameters
epochs = 40
train_loss = []
val_loss = []
val_acc = []
best_acc = 0
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    mlp_head.train()
    train_losses = []
    with tqdm(train_loader, desc="Train") as pbar:
        for i, (features, labels) in enumerate(pbar):
            features, labels = features.to(device, dtype=torch.float), labels.to(device)
            pred = mlp_head(features)
            loss = loss_fn(pred.type(torch.float), labels.unsqueeze(1).type(torch.float))
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            pbar.set_postfix(loss=np.mean(train_losses))

    # Each epoch, save the training loss to a CSV file
    path = root_dir / 'results/mlp_train.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in train_loss:
            writer.writerow([l])

    mlp_head.eval()
    valLoss, valAcc = validation(val_loader, mlp_head, loss_fn, device)
    val_loss.append(valLoss)
    val_acc.append(valAcc)

    # Save the best performing model
    if valAcc > best_acc:
        best_acc = valAcc
        dict_path = root_dir / 'model/best_mlp.pth'
        torch.save(mlp_head.state_dict(), dict_path)

    # Save the validation loss and accuracy
    with open(root_dir / 'results/mlp_validation.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in range(len(val_loss)):
            writer.writerow([val_loss[l], val_acc[l]])
