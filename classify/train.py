import os
import json
import keras
import models
import argparse
import numpy as np
from evaluate import evaluate
from sklearn.utils import compute_class_weight
from dataset import load_stratified_kfold, load_dataset_from_list


class Config:
    def __init__(self, args: argparse.Namespace):
        self.input_dir: str = args.input_dir
        self.output_dir: str = args.output_dir
        self.image_size: tuple[int, int] = args.image_size
        self.k_fold: int = args.k_fold
        self.batch_size: int = args.batch_size
        self.random_state: int = args.random_state
        self.epochs: int = args.epochs
        self.model_name: str = args.model
        self.num_classes: int = 0  # To be set after loading dataset

    def __str__(self) -> str:
        return (
            f"\t- Input Directory: {self.input_dir}\n"
            f"\t- Output Directory: {self.output_dir}\n"
            f"\t- Image Size: {self.image_size}\n"
            f"\t- K-Fold: {self.k_fold}\n"
            f"\t- Batch Size: {self.batch_size}\n"
            f"\t- Random State: {self.random_state}\n"
            f"\t- Epochs: {self.epochs}\n"
            f"\t- Model Name: {self.model_name}\n"
            f"\t- Number of Classes: {self.num_classes}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for training a classification model with k-fold cross-validation"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to the input directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--image_size",
        type=tuple,
        default=(224, 224),
        help="Size of the input images (default: (224, 224))",
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of epochs for training"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=models.models_dict.keys(),
        required=True,
        help="Model architecture to use",
    )
    args = parser.parse_args()

    # Initialize configuration
    config = Config(args)

    # Read dataset in input_dir
    X_train, y_train, X_test, y_test, info = load_stratified_kfold(
        config.input_dir, config.k_fold, config.random_state
    )
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    config.num_classes = len(info["label"])
    print(f">>> Configuration:\n{config}")

    print(f">>> Found {len(info['label'])} classes: {info['label']}")
    print(f"\t >>> Found {len(X_train)} training images")
    print(f"\t >>> Found {len(X_test)} testing images")

    # Prepare inputs
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights)}
    class_map = info["label_map"]
    print(f">>> Classes map: {class_map}")
    print(f">>> Class weights: {class_weights}")

    # Create output dir
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, config.model_name), exist_ok=True)

    # Get model
    print(f"\n>>> Using model: {config.model_name}")
    model: keras.Model = models.models_dict[config.model_name](
        image_size=config.image_size, num_classes=config.num_classes
    )

    # model.summary()

    # Data Augmentation
    data_augmentation_list = [
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.1),
        keras.layers.RandomContrast(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
    ]

    # Train
    results = []

    test_ds = load_dataset_from_list(
        X_test,
        y_test,
        config.num_classes,
        config.image_size,
        config.batch_size,
        is_test_set=True,
    )
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    for fold, (train_idx, val_idx) in enumerate(info["train_splits"]):
        model: keras.Model = models.models_dict[config.model_name](
            image_size=config.image_size, num_classes=config.num_classes
        )
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        train_ds = load_dataset_from_list(
            X_train_fold,
            y_train_fold,
            config.num_classes,
            config.image_size,
            config.batch_size,
            img_augmentation_layers=data_augmentation_list,
        )
        val_ds = load_dataset_from_list(
            X_val_fold,
            y_val_fold,
            config.num_classes,
            config.image_size,
            config.batch_size,
            is_test_set=True,
        )

        print(f">>> Training fold {fold + 1}/{config.k_fold}")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            class_weight=class_weights,
        )

        print(f">>> Evaluating fold {fold + 1}/{config.k_fold}")
        results.append(evaluate(test_ds, model, class_map))

        save_path = (
            f"{os.path.join(config.output_dir, config.model_name)}/fold-{fold}.keras"
        )
        model.save(save_path)
        print(f">>> Saved model to {save_path}")

    with open(
        f"{os.path.join(config.output_dir, config.model_name)}/results.json", "w"
    ) as f:
        json.dump(results, f)
    print(
        f">>> Saved results to {os.path.join(config.output_dir, config.model_name)}/results.json"
    )
