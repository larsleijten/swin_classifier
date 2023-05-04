import argparse
import sys
from typing import Union

sys.path.append("/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/")

import csv
import os
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# Import created classes
from swin_classifier.code.dataset import patchDataset
from swin_classifier.model.CombinedModel import CombinedModel
from swin_classifier.model.MLPhead import MLPhead
from swin_classifier.model.SwinEncoder import SwinEncoder


def main(
    root_dir: Union[Path, str] = "/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier"
) -> None:
    root_dir = Path(root_dir)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

    # Create the dataset that loads the patches
    dataset = patchDataset(root_dir / "data/ct_images/patches", transforms)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    swin_encoder = SwinEncoder(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=True,
    ).to(device, dtype=torch.float)

    # Create the MLP-head
    mlp_head = MLPhead(20736).to(device, dtype=torch.float)

    # Combine the two models make sure all parameters are training
    model = CombinedModel(swin_encoder, mlp_head).to(device, dtype=torch.float)
    model.load_state_dict(torch.load(root_dir / "model/best_combined_model.pth"))

    # No need to do dropout or batchnorm
    model.eval()

    with torch.no_grad(), open(root_dir / "data/features.csv", mode="w") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["PatchID", "label", "prediction", "feature_vector"])

        # Create features
        patch_id = 0
        for batch, label in tqdm(data_loader):
            batch = batch[None, :].to(device=device, dtype = torch.half)
            with torch.cuda.amp.autocast():
                feature_vector = swin_encoder(batch)
                prediction = mlp_head(feature_vector)

            # Save the features, prediction and labels
            for i in range(len(prediction)):
                writer.writerow([patch_id, label[i], prediction[i], feature_vector[i]])
                patch_id += 1


if __name__ == "__main__":
    # argparse for root folder
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier",
                        help="root directory of the project")
    args = parser.parse_args()

    main(
        root_dir=args.root_dir
    )
