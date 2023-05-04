import sys
sys.path.append('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/')

import tqdm
import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from torchvision.transforms import Compose, ToTensor, Normalize
import torch

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

transforms = Compose([
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

# Create the dataset that loads the patches
dataset = patchDataset('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/patches', transforms)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

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
model.load_state_dict(torch.load("/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/best_combined_model.pth"))
params = model.swin_encoder.state_dict()

# Create the SwinEncoder which will encode the patches
model = SwinEncoder(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
).to(device)

# Load pretrained weights

model.load_state_dict(params)
print("Using pretrained self-supervied Swin UNETR backbone weights !")

# No need to track gradients
model.eval()

# Create a file which saves the labels of the features
with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/feature_labels.csv', mode='w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["PatchID", "label"])

# Create features
patch_id = 0
for batch, label in tqdm(data_loader):
    batch = batch[None, :].to(device=device, dtype = torch.half)
    with torch.cuda.amp.autocast():
        feature_vector = model(batch)
    
    
    # Add the feature label
    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/feature_labels.csv', mode='a', newline='') as output_file:
                    writer = csv.writer(output_file)
                    writer.writerow([patch_id, int(label[0])])
    
    # Save the feature vector
    feature_path = os.path.join('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/features', str(patch_id) + ".pt")
    torch.save(feature_vector, feature_path)


    patch_id+=1

