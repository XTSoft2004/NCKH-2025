import shutil
from os import path as osp
from tqdm import tqdm
import os, json, argparse
from sklearn.model_selection import train_test_split


def is_image_file(filename: str) -> bool:
    """Check if the file is an image based on its extension."""
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for splitting datasets into training and testing sets. Simultaneously, script converts annotations from labelMe format to PaddleOCR format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the input images and annotations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output images and annotations will be saved.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of the dataset to be used for training (default: 0.8).",
    )
    args = parser.parse_args()

    # Ensure the input and output directories exist
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    assert osp.exists(args.input_dir)

    # Get all image files in the input directory
    image_names = list(filter(is_image_file, os.listdir(args.input_dir)))

    # Split the dataset into training and testing sets
    train_image_names, test_image_names = train_test_split(
        image_names,
        train_size=args.train_ratio,
        random_state=42,
        shuffle=True,
    )

    def process_images(image_names: list[str], prefix: str):
        output_image_dir = osp.join(args.output_dir, f"{prefix}_images")
        output_annotation_file = osp.join(args.output_dir, f"{prefix}_labels.txt")

        # Create the output directory for the current prefix if it doesn't exist
        if not osp.exists(output_image_dir):
            os.makedirs(output_image_dir)

        paddle_annotations = []
        for image_name in tqdm(image_names, desc=f">>> Processing {prefix} images"):
            image_name_no_ext = osp.splitext(image_name)[0]
            image_path = osp.join(args.input_dir, image_name)
            annotation_path = osp.join(args.input_dir, f"{image_name_no_ext}.json")

            # Check if the annotation file exists
            if not osp.exists(annotation_path):
                print(f"Annotation file {annotation_path} does not exist, skipping.")
                continue

            # Convert the annotation from labelMe format to PaddleOCR format
            with open(annotation_path, "r") as f:
                annotation = json.load(f)

            converted_annotation = [
                {
                    "transcription": shape.get("label", "###"),
                    "points": shape.get("points", []),
                }
                for shape in annotation.get("shapes", [])
            ]
            paddle_annotations.append(
                f"{osp.join(f"{prefix}_images", image_name)}\t{json.dumps(converted_annotation, ensure_ascii=False)}"
            )

            # Copy the image to the output directory
            shutil.copy(image_path, osp.join(output_image_dir, image_name))

        # Write the PaddleOCR formatted annotations to the output file
        with open(output_annotation_file, "w", encoding="utf-8") as f:
            f.write("\n".join(paddle_annotations))

    process_images(train_image_names, "train")
    process_images(test_image_names, "test")

    print(">>> Dataset split and conversion completed successfully.")
