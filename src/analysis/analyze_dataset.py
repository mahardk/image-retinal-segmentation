from pathlib import Path

from PIL import Image


def analyze_directory(directory):

    directory = Path(directory)

    image_files = []

    for extension in (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.tif",
        "*.tiff",
        "*.gif",
    ):
        image_files.extend(
            directory.rglob(extension)
        )

    print(
        f"Directory: {directory}"
    )

    print(
        f"Total files: {len(image_files)}"
    )

    for path in sorted(image_files):

        try:

            with Image.open(path) as image:

                print(
                    f"{path} | "
                    f"size={image.size} | "
                    f"mode={image.mode}"
                )

        except Exception as exc:

            print(
                f"ERROR: {path} | {exc}"
            )


if __name__ == "__main__":

    analyze_directory(
        "data/raw"
    )