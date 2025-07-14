import os
import argparse
import cv2
from tqdm import tqdm
import albumentations as A


class Config:
    def __init__(self, args: argparse.Namespace):
        if not os.path.exists(args.output_dir):
            os.makedirs(self.output_dir)
        self.__check_args(args)

        self.input_dir: str = args.input_dir
        self.output_dir: str = args.output_dir
        self.augmentation_config: str = args.augmentation_config
        self.augmentation_radio: int = args.augmentation_radio

    def __str__(self):
        return (
            f"\t- Input Directory: {self.input_dir}\n"
            f"\t- Output Directory: {self.output_dir}\n"
            f"\t- Augmentation Configuration: {self.augmentation_config}\n"
            f"\t- Augmentation Radio: {self.augmentation_radio}"
        )

    def __check_args(self, args: argparse.Namespace):
        assert os.path.exists(args.input_dir), "Input directory does not exist"
        assert os.path.exists(
            args.augmentation_config
        ), "!!! Augmentation configuration file does not exist"
        assert (
            args.augmentation_radio > 0
        ), "!!! Augmentation radio must be a positive integer"
        assert os.path.isdir(args.input_dir), "!!! Input path must be a directory"
        assert os.path.isdir(args.output_dir), "!!! Output path must be a directory"
        assert args.augmentation_config.endswith(
            ".yaml"
        ), "!!! Augmentation configuration must be a YAML file"
        assert (
            args.input_dir != args.output_dir
        ), "!!! Input and output directories must be different"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for data augmentation")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing input data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save augmented data"
    )
    parser.add_argument(
        "--augmentation_config",
        type=str,
        required=True,
        help="Path to the augmentation configuration file (YAML format)",
    )
    parser.add_argument(
        "--augmentation_radio",
        type=int,
        default=3,
        help="Number of augmentations to apply per image (default: 3). This is used to control the number of augmented images generated per original image.",
    )
    args = parser.parse_args()

    config = Config(args)
    print(">>> Configuration:")
    print(config)

    transform = A.load(config.augmentation_config, data_format="yaml")
    print(f">>> Loaded augmentation configuration from {config.augmentation_config}")
    print(f"\t{transform}")

    input_images = list(
        filter(
            lambda x: x.endswith((".jpg", ".jpeg", ".png")),
            os.listdir(config.input_dir),
        )
    )
    for image_name in tqdm(input_images, ascii=True, desc=">>> Validating images"):
        image_name_without_ext = os.path.splitext(image_name)[0]
        annotation_path = os.path.join(
            config.input_dir, f"{image_name_without_ext}.json"
        )
        assert os.path.exists(
            annotation_path
        ), f"!!! Annotation file {annotation_path} does not exist for image {image_name}."
    assert 2 * len(input_images) == len(
        os.listdir(config.input_dir)
    ), "!!! Input directory must contain images and their corresponding annotations."

    print(f">>> Found {len(list(input_images))} images in the input directory")

    for image_name in tqdm(input_images, ascii=True, desc=">>> Data augmentation"):
        image_path = os.path.join(config.input_dir, image_name)
        image_name_without_ext = os.path.splitext(image_name)[0]
        annotation_path = os.path.join(
            config.input_dir, f"{image_name_without_ext}.json"
        )

        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

            output_image_path = os.path.join(config.output_dir, image_name)
            output_annotation_path = os.path.join(
                config.output_dir, f"{image_name_without_ext}.json"
            )
            cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # type: ignore
            with open(annotation_path, "r") as f:
                annotation_data = f.read()
            with open(output_annotation_path, "w") as f:
                f.write(annotation_data)

            for i in range(config.augmentation_radio):
                transformed = transform(image=image)["image"]  # type: ignore

                output_image_name = f"{image_name_without_ext}_aug_{i + 1}.jpg"
                output_annotation_name = f"{image_name_without_ext}_aug_{i + 1}.json"
                output_image_path = os.path.join(config.output_dir, output_image_name)
                output_annotation_path = os.path.join(
                    config.output_dir, output_annotation_name
                )

                cv2.imwrite(
                    output_image_path, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
                )
                with open(annotation_path, "r") as f:
                    annotation_data = f.read()
                with open(output_annotation_path, "w") as f:
                    f.write(annotation_data)

        except Exception as e:
            print(f"!!! Error reading image {image_name}: {e}")
            continue
