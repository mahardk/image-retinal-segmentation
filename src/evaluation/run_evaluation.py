from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import build_test_records, read_mask, read_rgb
from src.evaluation.evaluate import calculate_metrics
from src.inference.predict import sliding_window_predict
from src.model.cas_unet import CASUNet
from src.preprocessing.pipeline import preprocess_image


def load_config(config_path: str = "configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_checkpoint(model, checkpoint_path: Path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return checkpoint


def evaluate_scenario(
    scenario: str,
    records,
    config,
    device,
):
    checkpoint_path = Path(
        f"results/checkpoints/best_{scenario}.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint_path}"
        )

    model = CASUNet(
        in_channels=config["model"]["input_channels"],
        out_channels=config["model"]["output_channels"],
        channels=tuple(config["model"]["channels"]),
        attention_reduction=config["model"]["attention_reduction"],
    )

    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    threshold = config["evaluation"]["threshold"]
    patch_size = config["patch"]["size"]

    all_metrics = []

    print()
    print("=" * 60)
    print(f"Scenario: {scenario}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    print("=" * 60)

    for index, record in enumerate(records, start=1):
        rgb_image = read_rgb(record.image_path)

        image = preprocess_image(
            rgb_image,
            scenario=scenario,
            clahe_clip_limit=config["preprocessing"]["clahe"]["clip_limit"],
            clahe_tile_grid_size=tuple(
                config["preprocessing"]["clahe"]["tile_grid_size"]
            ),
            gamma=config["preprocessing"]["gamma"]["value"],
        )

        if record.vessel_gt_path is None:
            print(
                f"[{index}/{len(records)}] "
                f"{record.image_path.name}: skipped, no vessel GT"
            )
            continue

        ground_truth = read_mask(record.vessel_gt_path)

        probability = sliding_window_predict(
            model=model,
            image=image,
            device=device,
            patch_size=patch_size,
            stride=patch_size,
        )

        prediction = probability >= threshold

        metrics = calculate_metrics(
            prediction=prediction,
            ground_truth=ground_truth,
        )

        row = {
            "Scenario": scenario,
            "Image": record.image_path.name,
            "Accuracy": metrics["Accuracy"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "F1-Score": metrics["F1-Score"],
            "IoU": metrics["IoU"],
        }

        all_metrics.append(row)

        print(
            f"[{index}/{len(records)}] "
            f"{record.image_path.name} | "
            f"Accuracy={metrics['Accuracy']:.4f} | "
            f"Sensitivity={metrics['Sensitivity']:.4f} | "
            f"Specificity={metrics['Specificity']:.4f} | "
            f"F1={metrics['F1-Score']:.4f} | "
            f"IoU={metrics['IoU']:.4f}"
        )

    if not all_metrics:
        raise RuntimeError(
            f"Tidak ada image yang berhasil dievaluasi untuk {scenario}."
        )

    dataframe = pd.DataFrame(all_metrics)

    metric_names = [
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "F1-Score",
        "IoU",
    ]

    averages = {
        metric: float(dataframe[metric].mean())
        for metric in metric_names
    }

    print()
    print(f"Average metrics - {scenario}")

    for metric in metric_names:
        print(f"{metric}: {averages[metric]:.4f}")

    return dataframe, averages


def save_charts(summary_dataframe, output_dir):
    metric_names = [
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "F1-Score",
        "IoU",
    ]

    # Chart semua metrik
    plt.figure(figsize=(12, 7))

    x = np.arange(len(summary_dataframe))
    width = 0.15

    for index, metric in enumerate(metric_names):
        plt.bar(
            x + (index - 2) * width,
            summary_dataframe[metric] * 100,
            width,
            label=metric,
        )

    plt.xticks(
        x,
        summary_dataframe["Scenario"],
    )
    plt.ylabel("Persentase (%)")
    plt.title("Perbandingan Hasil Segmentasi")
    plt.ylim(50, 100)
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.2,
    )
    plt.tight_layout()

    plt.savefig(
        output_dir / "metrics_comparison.png",
        dpi=200,
    )
    plt.close()

    # Chart F1 dan IoU
    plt.figure(figsize=(10, 6))

    x = np.arange(len(summary_dataframe))
    width = 0.35

    plt.bar(
        x - width / 2,
        summary_dataframe["F1-Score"] * 100,
        width,
        label="F1-Score",
    )

    plt.bar(
        x + width / 2,
        summary_dataframe["IoU"] * 100,
        width,
        label="IoU",
    )

    plt.xticks(
        x,
        summary_dataframe["Scenario"],
    )
    plt.ylabel("Persentase (%)")
    plt.title("Perbandingan F1-Score dan IoU")
    plt.ylim(50, 85)
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.2,
    )
    plt.tight_layout()

    plt.savefig(
        output_dir / "f1_iou_comparison.png",
        dpi=200,
    )
    plt.close()


def main():
    config = load_config()

    output_dir = Path("results/evaluation")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = config["training"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA tidak tersedia. Menggunakan CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Device: {device}")

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

    print(f"Total test records: {len(records)}")
    print(f"CHASE_DB1 test records: {len(chase_records)}")

    scenarios = [
        "grayscale",
        "clahe",
        "clahe_gamma",
    ]

    all_results = []
    summary_results = []

    for scenario in scenarios:
        dataframe, averages = evaluate_scenario(
            scenario=scenario,
            records=chase_records,
            config=config,
            device=device,
        )

        all_results.append(dataframe)

        summary_row = {
            "Scenario": scenario,
            **averages,
        }

        summary_results.append(summary_row)

    per_image_dataframe = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary_dataframe = pd.DataFrame(
        summary_results
    )

    per_image_path = (
        output_dir / "evaluation_per_image.csv"
    )

    summary_path = (
        output_dir / "evaluation_summary.csv"
    )

    per_image_dataframe.to_csv(
        per_image_path,
        index=False,
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    save_charts(
        summary_dataframe,
        output_dir,
    )

    print()
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        summary_dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("=" * 60)
    print("FILES GENERATED")
    print("=" * 60)
    print(f"Per-image results : {per_image_path}")
    print(f"Summary results   : {summary_path}")
    print(
        f"Metrics chart     : "
        f"{output_dir / 'metrics_comparison.png'}"
    )
    print(
        f"F1/IoU chart      : "
        f"{output_dir / 'f1_iou_comparison.png'}"
    )


if __name__ == "__main__":
    main()