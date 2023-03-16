import sys
sys.path.append('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/')


import tqdm
import os
import random
import csv
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
from monai.config import print_config

# Import created classes
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
random.seed(64)
np.random.seed(64)
torch.manual_seed(64)

transforms = Compose([
    ToTensor(), 
    Normalize(mean=[0.5], std=[0.5])
])

# Create dataset and split in train and validation
dataset = patchDataset('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/patches', transforms)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# Create the Swin Encoder and load the pretrained weights
swin_encoder = SwinEncoder(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
).to(device, dtype=torch.float)

swin_weights = torch.load("/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/model_swinvit.pt")
swin_encoder.load_from(weights=swin_weights)

# Create the MLP-head
mlp_head = MLPhead(20736).to(device, dtype=torch.float)


# Combine the two models make sure all parameters are training
model = CombinedModel(swin_encoder, mlp_head).to(device, dtype=torch.float)
#model.load_state_dict(torch.load("/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/best_combined_model.pth"))
for param in model.parameters():
    param.requires_grad = True


# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss().to(device)

epochs = 50
model.train()
train_loss = []
val_loss = []
val_acc = []
best_acc = 0
total_loss = 0

# Start training
for t in range(epochs):
    print("Epoch number: " + str(t))
    #for param in model.parameters():
        # To prevent NaN outputs
        #param = torch.clamp(param, min = -10.0, max = 10.0)
    for i, (batch, labels) in enumerate(tqdm(train_loader)):
        batch, labels = batch[None, :].to(device=device, dtype = torch.float), labels.to(device)
        # Call model
        with torch.cuda.amp.autocast():
            pred = model(batch)        
            # Calculate loss and backpropagate
            loss = loss_fn(pred, torch.tensor(labels, dtype=torch.half))
        
        # Calculate loss and backpropagate
        total_loss = total_loss + loss.item()
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        #nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track training loss
        if i % 50 ==0:
            train_loss.append(total_loss / 50)
            total_loss = 0

    # Save the training loss each epoch
    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/combined_train.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in train_loss:
            writer.writerow([l])
    
    # Perform validation after each epoch
    model.eval()
    valLoss, valAcc = validation(val_loader, model, loss_fn, device)
    model.train()
    val_loss.append(valLoss)
    val_acc.append(valAcc)

    # Save the best performing model
    if valAcc > best_acc:
        best_acc = valAcc
        dict_path = '/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/best_combined_model.pth'
        torch.save(model.state_dict(), dict_path)

    # Save the validation loss and 
    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/combined_validation.csv', mode='w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        for l in range(len(val_loss)):
            writer.writerow([val_loss[l], val_acc[l]])
    


# Also save the latest epoch of the training, in case of interrupted training
dict_path = '/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/last_combined_model.pth'
torch.save(model.state_dict(), dict_path)


