import monai
import nibabel
import tqdm
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

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
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch

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

model.eval()



mlp_head = MLPhead(feature_vector.size(dim=0)).to(device)

i=0
for batch, labels in val_loader:
    batch = batch[None, :].to(device=device, dtype = torch.half)
    print(batch.size())
    with torch.cuda.amp.autocast():
        results = model(batch)
    print(results.size()) # batch size will be (32, z, x, y)
    print(labels.size())
    if (i>5):
        break
    i+=1



