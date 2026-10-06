import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, random_split

from src.data.dataset import (
    build_training_records,
    read_mask,
    read_rgb,
)

from src.patches.balanced_sampling import (
    build_balanced_patch_index,
)

from src.preprocessing.pipeline import (
    preprocess_image,
)

from src.model.cas_unet import CASUNet
from src.training.losses import BCEDiceLoss


class PatchDataset(Dataset):

    def __init__(
        self,
        records,
        patch_locations,
        scenario,
        patch_size,
    ):
        self.records = records
        self.locations = patch_locations
        self.scenario = scenario
        self.patch_size = patch_size

        self.cache = {}

    def _load_record(self, index):

        if index not in self.cache:

            record = self.records[index]

            rgb = read_rgb(
                record.image_path
            )

            image = preprocess_image(
                rgb,
                scenario=self.scenario,
            )

            mask = read_mask(
                record.vessel_gt_path
            )

            self.cache[index] = (
                image,
                mask,
            )

        return self.cache[index]

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):

        location = self.locations[idx]

        image, mask = self._load_record(
            location.record_index
        )

        y = location.y
        x = location.x

        image_patch = image[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        mask_patch = mask[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        image_patch = (
            torch.from_numpy(
                image_patch
            ).float()
            / 255.0
        )

        mask_patch = torch.from_numpy(
            mask_patch
        ).float()

        image_patch = image_patch.unsqueeze(0)
        mask_patch = mask_patch.unsqueeze(0)

        return image_patch, mask_patch


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

    return total_loss / len(loader.dataset)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        required=True,
        choices=[
            "grayscale",
            "clahe",
            "clahe_gamma",
        ],
    )

    args = parser.parse_args()

    with open(
        "configs/config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    set_seed(
        config["seed"]
    )

    device = torch.device(
        "cuda"
        if (
            config["training"]["device"] == "cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    records = build_training_records(
        config["dataset"]["drive_dir"],
        config["dataset"]["chase_dir"],
        config["dataset"]["chase_train_subjects"],
    )

    print(
        "Training images:",
        len(records),
    )

    locations = build_balanced_patch_index(
        records=records,
        patch_size=config["patch"]["size"],
        patches_per_image=config["patch"]["patches_per_image"],
        seed=config["seed"],
    )

    print(
        "Total patches:",
        len(locations),
    )

    dataset = PatchDataset(
        records=records,
        patch_locations=locations,
        scenario=args.scenario,
        patch_size=config["patch"]["size"],
    )

    val_size = int(
        len(dataset)
        * config["training"]["validation_split"]
    )

    train_size = (
        len(dataset)
        - val_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            config["seed"]
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = CASUNet(
        in_channels=config["model"]["input_channels"],
        out_channels=config["model"]["output_channels"],
        channels=tuple(
            config["model"]["channels"]
        ),
        attention_reduction=config["model"]["attention_reduction"],
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    checkpoint_dir = Path(
        "results/checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(
        1,
        config["training"]["epochs"] + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scenario": args.scenario,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_dir
                / f"best_{args.scenario}.pth",
            )


if __name__ == "__main__":
    main()