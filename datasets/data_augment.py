import cv2, os
import os.path as osp
import albumentations as A

IMG_HEIGHT, IMG_WIDTH = 640, 640

transform = A.Compose(
    [
        A.SomeOf(
            n=3,
            transforms=[
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0, 2.0)),
                A.MotionBlur(blur_limit=(3, 5)),
                A.GaussNoise(std_range=(0.02, 0.05), mean_range=(0, 0)),
                A.CoarseDropout(
                    num_holes_range=(5, 15),
                    hole_height_range=(10, int(IMG_HEIGHT * 0.03)),
                    hole_width_range=(10, int(IMG_WIDTH * 0.03)),
                    fill=0,
                ),
            ],
            p=1.0,
        )
    ]
)

DATASET_INPUT_DIR = osp.join("final", "datasets-2", "student_card", "train_images")
LABEL_INPUT_PATH = osp.join("final", "datasets-2", "student_card", "train_labels.txt")
DATASET_OUTPUT_DIR = osp.join("final", "datasets-2-aug", "student_card", "train_images")
LABEL_OUTPUT_PATH = osp.join("final", "datasets-2-aug", "train_labels.txt")

if not osp.exists(DATASET_OUTPUT_DIR):
    os.makedirs(DATASET_OUTPUT_DIR)

labels_map = dict()
with open(LABEL_INPUT_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.split("\t")
        labels_map[line[0]] = line[1]

labels_aug_map = dict()
for image_name in os.listdir(DATASET_INPUT_DIR):
    image_name_without_ext = osp.splitext(image_name)[0]
    image_ext = osp.splitext(image_name)[1]
    image_path = osp.join(DATASET_INPUT_DIR, image_name)

    image = cv2.imread(image_path)
    cv2.imwrite(osp.join(DATASET_OUTPUT_DIR, image_name), image)
    labels_aug_map[image_name] = labels_map[image_name]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(3):
        augmented = transform(image=image)
        augmented_image = augmented["image"]
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

        labels_aug_map[f"{image_name_without_ext}_aug_{i}{image_ext}"] = labels_map[
            image_name
        ]
        cv2.imwrite(
            osp.join(
                DATASET_OUTPUT_DIR, f"{image_name_without_ext}_aug_{i}{image_ext}"
            ),
            augmented_image,
        )

with open(LABEL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    for key in labels_aug_map.keys():
        content = f"{key}\t{labels_aug_map[key]}"
        if not content.endswith("\n"):
            content += "\n"
        f.write(content)

print("Data augmentation completed.")
