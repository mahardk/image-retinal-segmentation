from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.dataset import (
    build_test_records,
    read_mask,
    read_rgb,
)
from src.inference.predict import sliding_window_predict
from src.model.cas_unet import CASUNet
from src.preprocessing.pipeline import preprocess_image


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(scenario, config, device):
    model = CASUNet(
        in_channels=config["model"]["input_channels"],
        out_channels=config["model"]["output_channels"],
        channels=tuple(config["model"]["channels"]),
        attention_reduction=config["model"]["attention_reduction"],
    )

    checkpoint_path = Path(
        f"results/checkpoints/best_{scenario}.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def main():
    config = load_config()

    output_dir = Path(
        "results/evaluation/predictions"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = config["training"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    records = build_test_records(
        drive_dir=config["dataset"]["drive_dir"],
        chase_dir=config["dataset"]["chase_dir"],
        chase_test_subjects=config["dataset"]["chase_test_subjects"],
    )

    chase_records = [
        record
        for record in records
        if record.dataset == "CHASE_DB1"
    ]

    # Menggunakan Image_11L sebagai contoh visual.
    record = next(
        record
        for record in chase_records
        if record.image_path.name == "Image_11L.jpg"
    )

    print("=" * 60)
    print("PREDICTION VISUALIZATION")
    print("=" * 60)
    print(f"Image : {record.image_path.name}")
    print(f"Device: {device}")

    rgb_image = read_rgb(record.image_path)
    ground_truth = read_mask(
        record.vessel_gt_path
    )

    scenarios = {
        "Grayscale": "grayscale",
        "CLAHE": "clahe",
        "CLAHE + Gamma 0.8": "clahe_gamma",
    }

    predictions = {}

    for display_name, scenario in scenarios.items():
        print(f"Processing: {display_name}")

        image = preprocess_image(
            rgb_image,
            scenario=scenario,
            clahe_clip_limit=config["preprocessing"]["clahe"]["clip_limit"],
            clahe_tile_grid_size=tuple(
                config["preprocessing"]["clahe"]["tile_grid_size"]
            ),
            gamma=config["preprocessing"]["gamma"]["value"],
        )

        model = load_model(
            scenario=scenario,
            config=config,
            device=device,
        )

        probability = sliding_window_predict(
            model=model,
            image=image,
            device=device,
            patch_size=config["patch"]["size"],
            stride=config["patch"]["size"],
        )

        prediction = (
            probability
            >= config["evaluation"]["threshold"]
        )

        predictions[display_name] = prediction

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 5),
    )

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original RGB")

    axes[1].imshow(
        ground_truth,
        cmap="gray",
    )
    axes[1].set_title("Ground Truth")

    for ax, (title, prediction) in zip(
        axes[2:],
        predictions.items(),
    ):
        ax.imshow(
            prediction,
            cmap="gray",
        )
        ax.set_title(title)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = (
        output_dir
        / "Image_11L_prediction_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("=" * 60)
    print("FILE GENERATED")
    print("=" * 60)
    print(output_path)


if __name__ == "__main__":
    main()