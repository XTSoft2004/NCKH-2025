import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


def load_stratified_kfold(input_dir: str, k: int, random_state: int = 42):
    """
    Loads image file paths and their corresponding class labels from a directory structure,
    and generates stratified K-fold splits for cross-validation.

    Args:
      input_dir (str): Path to the root directory containing subdirectories for each class.
      k (int): Number of folds for stratified K-fold cross-validation.
      random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
      tuple: A tuple containing:
        - splits (list): List of (train_idx, test_idx) tuples for each fold.
        - X (list): List of image file paths.
        - y (list): List of class labels corresponding to each image.
        - classes (list): List of class names found in the input directory.
    """
    classes = os.listdir(input_dir)

    X: list[str] = []
    y: list[int] = []

    class_map = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        for img_name in os.listdir(cls_dir):
            X.append(os.path.join(cls_dir, img_name))
            y.append(class_map[cls])

    kf = StratifiedKFold(n_splits=k, random_state=random_state, shuffle=True)
    return list(kf.split(X, y)), X, y, class_map


def load_dataset(
    image_paths: list[str] | np.ndarray,
    labels: list | np.ndarray,
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
):
    """
    Loads and preprocesses images and labels into a TensorFlow dataset.

    Args:
      image_paths (list[str] | np.ndarray): List of file paths to the images.
      labels (list | np.ndarray): List of labels corresponding to each image.
      image_size (tuple[int, int], optional): Target size to resize images (height, width). Defaults to (224, 224).
      batch_size (int, optional): Number of samples per batch. Defaults to 32.

    Returns:
      tf.data.Dataset: A TensorFlow dataset yielding batches of preprocessed images and their labels.
    """

    def _load_and_preprocess(path: str, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # type: ignore
        img = tf.image.resize(img, image_size)
        return img, tf.one_hot(label, depth=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
