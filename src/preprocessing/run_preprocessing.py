from pathlib import Path

import cv2

from src.data.dataset import read_rgb
from src.preprocessing.grayscale import rgb_to_grayscale


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRIVE_TRAINING = PROJECT_ROOT / "data" / "raw" / "DRIVE" / "training" / "images"
DRIVE_TEST = PROJECT_ROOT / "data" / "raw" / "DRIVE" / "test" / "images"
CHASE = PROJECT_ROOT / "data" / "raw" / "CHASEDB1"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "preprocessing" / "grayscale"


def process_folder(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".png",
    }

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    print(f"\nInput : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(image_paths)}")

    for image_path in image_paths:

        rgb = read_rgb(image_path)

        gray = rgb_to_grayscale(rgb)

        output_path = output_dir / f"{image_path.stem}_gray.png"

        success = cv2.imwrite(
            str(output_path),
            gray,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save: {output_path}"
            )

        print(f"[OK] {image_path.name} -> {output_path.name}")


def main():
    print("=" * 60)
    print("GRAYSCALE PREPROCESSING")
    print("=" * 60)

    process_folder(
        DRIVE_TRAINING,
        OUTPUT_ROOT / "DRIVE" / "training",
    )

    process_folder(
        DRIVE_TEST,
        OUTPUT_ROOT / "DRIVE" / "test",
    )

    process_folder(
        CHASE,
        OUTPUT_ROOT / "CHASEDB1",
    )

    print("\n" + "=" * 60)
    print("GRAYSCALE PREPROCESSING SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()