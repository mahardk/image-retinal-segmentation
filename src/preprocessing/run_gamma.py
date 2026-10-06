from pathlib import Path

import cv2

from src.preprocessing.gamma import apply_gamma


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "clahe"
    / "DRIVE"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "clahe_gamma"
    / "DRIVE"
)


def process_folder(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

    print(f"\nInput : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(image_paths)}")

    for image_path in image_paths:
        gray_clahe = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray_clahe is None:
            raise RuntimeError(
                f"Failed to read: {image_path}"
            )

        gamma_image = apply_gamma(
            gray_clahe,
            gamma=0.8,
        )

        output_path = (
            output_dir
            / f"{image_path.stem}_gamma.png"
        )

        success = cv2.imwrite(
            str(output_path),
            gamma_image,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save: {output_path}"
            )

        print(
            f"[OK] {image_path.name} "
            f"-> {output_path.name}"
        )


def main():
    print("=" * 60)
    print("GAMMA CORRECTION PREPROCESSING")
    print("Gamma = 0.8")
    print("=" * 60)

    process_folder(
        INPUT_ROOT / "training",
        OUTPUT_ROOT / "training",
    )

    process_folder(
        INPUT_ROOT / "test",
        OUTPUT_ROOT / "test",
    )

    print("\n" + "=" * 60)
    print("GAMMA CORRECTION SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()