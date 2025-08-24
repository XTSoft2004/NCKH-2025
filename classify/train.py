import os
import json
import keras
import models
import argparse
import numpy as np
from evaluate import evaluate
from dataset import load_dataset, load_stratified_kfold
from sklearn.utils.class_weight import compute_class_weight


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
        self.num_classes: int = len(os.listdir(args.input_dir))

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
        default=32,
        help="Batch size for training (default: 32)",
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
    print(f">>> Configuration:\n{config}")

    # Create output dir
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, config.model_name), exist_ok=True)

    # Get model
    print(f">>> Using model: {config.model_name}")
    model: keras.Model = models.models_dict[config.model_name](
        image_size=config.image_size, num_classes=config.num_classes
    )
    model.summary()

    # Load dataset and split with Stratified K-Fold
    splits, X, y, class_map = load_stratified_kfold(
        input_dir=config.input_dir, k=config.k_fold, random_state=config.random_state
    )
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}
    print(f">>> Classes map: {class_map}")
    print(f">>> Class weights: {class_weights}")

    # Save class map
    with open(
        f"{os.path.join(config.output_dir, config.model_name)}/class_map.json", "w"
    ) as f:
        json.dump(class_map, f)

    # Save class weights
    with open(
        f"{os.path.join(config.output_dir, config.model_name)}/class_weights.json", "w"
    ) as f:
        json.dump(class_weights, f)

    X = np.array(X)
    y = np.array(y)
    results = []

    # Train
    for fold, (train_idx, val_idx) in enumerate(splits):
        model: keras.Model = models.models_dict[config.model_name](
            image_size=config.image_size, num_classes=config.num_classes
        )
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        X_train_paths = X[train_idx]
        y_train = y[train_idx]

        X_test_paths = X[val_idx]
        y_test = y[val_idx]

        train_ds = load_dataset(
            X_train_paths,
            y_train,
            config.num_classes,
            config.image_size,
            config.batch_size,
        )

        print(f">>> Training fold {fold + 1}/{config.k_fold}")
        model.fit(train_ds, epochs=config.epochs, class_weight=class_weights)

        print(f">>> Evaluating fold {fold + 1}/{config.k_fold}")
        test_ds = load_dataset(
            X_test_paths,
            y_test,
            config.num_classes,
            config.image_size,
            config.batch_size,
        )
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
