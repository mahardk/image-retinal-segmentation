
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from src.data.dataset import build_test_records, read_rgb
from src.preprocessing.pipeline import preprocess_image


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_statistics(image):
    return {
        "Minimum": int(image.min()),
        "Maximum": int(image.max()),
        "Mean": float(image.mean()),
        "Median": float(np.median(image)),
        "Std": float(image.std()),
    }


def save_histogram(image, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        image.ravel(),
        bins=256,
        range=(0, 256),
        color="steelblue",
    )

    ax.set_title(f"Histogram Intensitas - {title}")
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Frekuensi")
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_histogram_comparison(images, output_path):
    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(18, 5),
    )

    for ax, (title, image) in zip(axes, images.items()):
        ax.hist(
            image.ravel(),
            bins=256,
            range=(0, 256),
            color="steelblue",
        )

        ax.set_title(f"Histogram\n{title}")
        ax.set_xlabel("Intensitas Piksel")
        ax.set_ylabel("Frekuensi")
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_image_comparison(images, output_path):
    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(18, 6),
    )

    for ax, (title, image) in zip(axes, images.items()):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    config = load_config()

    output_dir = Path("results/preprocessing/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if not chase_records:
        raise RuntimeError(
            "Tidak ditemukan citra test CHASE_DB1. "
            "Periksa dataset dan konfigurasi."
        )

    # Menggunakan citra yang sama untuk ketiga skenario.
    record = chase_records[0]
    rgb_image = read_rgb(record.image_path)

    scenarios = {
        "Grayscale": {
            "scenario": "grayscale",
            "filename": "01_grayscale.png",
            "histogram": "01_histogram_grayscale.png",
        },
        "Grayscale + CLAHE": {
            "scenario": "clahe",
            "filename": "02_grayscale_clahe.png",
            "histogram": "02_histogram_grayscale_clahe.png",
        },
        "Grayscale + CLAHE + Gamma 0.8": {
            "scenario": "clahe_gamma",
            "filename": "03_grayscale_clahe_gamma_08.png",
            "histogram": "03_histogram_grayscale_clahe_gamma_08.png",
        },
    }

    processed_images = {}
    statistics = []

    print("=" * 65)
    print("PREPROCESSING ANALYSIS")
    print("=" * 65)
    print(f"Dataset : {record.dataset}")
    print(f"Image   : {record.image_path.name}")
    print(f"Path    : {record.image_path}")
    print(f"Shape   : {rgb_image.shape}")
    print()

    for display_name, details in scenarios.items():
        image = preprocess_image(
            rgb_image,
            scenario=details["scenario"],
            clahe_clip_limit=config["preprocessing"]["clahe"]["clip_limit"],
            clahe_tile_grid_size=tuple(
                config["preprocessing"]["clahe"]["tile_grid_size"]
            ),
            gamma=config["preprocessing"]["gamma"]["value"],
        )

        processed_images[display_name] = image

        # Simpan gambar hasil preprocessing secara terpisah.
        image_path = output_dir / details["filename"]
        Image.fromarray(image).save(image_path)

        # Simpan histogram masing-masing hasil.
        histogram_path = output_dir / details["histogram"]
        save_histogram(
            image=image,
            title=display_name,
            output_path=histogram_path,
        )

        # Hitung statistik intensitas.
        stats = calculate_statistics(image)
        stats["Scenario"] = display_name
        statistics.append(stats)

        print(display_name)
        print(f"  Shape   : {image.shape}")
        print(f"  Min     : {stats['Minimum']}")
        print(f"  Max     : {stats['Maximum']}")
        print(f"  Mean    : {stats['Mean']:.2f}")
        print(f"  Median  : {stats['Median']:.2f}")
        print(f"  Std     : {stats['Std']:.2f}")
        print(f"  Image   : {image_path}")
        print(f"  Hist    : {histogram_path}")
        print()

    # Gabungkan ketiga gambar preprocessing.
    comparison_path = (
        output_dir / "04_preprocessing_comparison.png"
    )
    save_image_comparison(
        images=processed_images,
        output_path=comparison_path,
    )

    # Gabungkan ketiga histogram dalam satu gambar.
    histogram_comparison_path = (
        output_dir / "04_histogram_comparison.png"
    )
    save_histogram_comparison(
        images=processed_images,
        output_path=histogram_comparison_path,
    )

    # Simpan statistik ke CSV.
    dataframe = pd.DataFrame(statistics)
    dataframe = dataframe[
        [
            "Scenario",
            "Minimum",
            "Maximum",
            "Mean",
            "Median",
            "Std",
        ]
    ]

    csv_path = output_dir / "preprocessing_statistics.csv"
    dataframe.to_csv(csv_path, index=False)

    print("=" * 65)
    print("SEMUA OUTPUT SELESAI DIBUAT")
    print("=" * 65)
    print(f"Image comparison     : {comparison_path}")
    print(f"Histogram comparison : {histogram_comparison_path}")
    print(f"Statistics CSV       : {csv_path}")
    print(f"Output directory     : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
