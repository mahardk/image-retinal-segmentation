from pathlib import Path

import cv2

from src.preprocessing.clahe import apply_clahe


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = PROJECT_ROOT / "results" / "preprocessing" / "grayscale"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "preprocessing" / "clahe"


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
        gray = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray is None:
            raise RuntimeError(
                f"Failed to read: {image_path}"
            )

        clahe = apply_clahe(gray)

        output_path = output_dir / f"{image_path.stem}_clahe.png"

        success = cv2.imwrite(
            str(output_path),
            clahe,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save: {output_path}"
            )

        print(f"[OK] {image_path.name} -> {output_path.name}")


def main():
    print("=" * 60)
    print("CLAHE PREPROCESSING")
    print("=" * 60)

    process_folder(
        INPUT_ROOT / "DRIVE" / "training",
        OUTPUT_ROOT / "DRIVE" / "training",
    )

    process_folder(
        INPUT_ROOT / "DRIVE" / "test",
        OUTPUT_ROOT / "DRIVE" / "test",
    )

    print("\n" + "=" * 60)
    print("CLAHE PREPROCESSING SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()