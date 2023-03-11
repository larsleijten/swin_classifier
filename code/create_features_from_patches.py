import monai
import nibabel
import tqdm
import os
import shutil
import tempfile
import csv

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from torchvision.transforms import Compose, ToTensor, Normalize
import torch

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



# create data loaders for the training and validation sets
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
).to(device)

weight = torch.load("/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/model/model_swinvit.pt")
model.load_from(weights=weight)
print("Using pretrained self-supervied Swin UNETR backbone weights !")

model.eval()

with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/feature_labels.csv', mode='w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["PatchID", "label"])

patch_id = 0
for batch, label in tqdm(data_loader):
    batch = batch[None, :].to(device=device, dtype = torch.half)
    with torch.cuda.amp.autocast():
        feature_vector = model(batch)
    
    
    
    with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/feature_labels.csv', mode='a', newline='') as output_file:
                    writer = csv.writer(output_file)
                    writer.writerow([patch_id, int(label[0])])
    
    feature_path = os.path.join('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/features', str(patch_id) + ".pt")
    torch.save(feature_vector, feature_path)


    patch_id+=1

