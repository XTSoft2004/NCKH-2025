import os
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (
    BaseImagePreprocessingLayer,
)


def load_stratified_kfold(input_dir: str, k: int, random_state: int = 42):
    info = {}
    info["label"] = os.listdir(input_dir)
    info["label_map"] = {label: i for i, label in enumerate(info["label"])}

    X_train: list[str] = []
    y_train: list[int] = []

    X_test: list[str] = []
    y_test: list[int] = []

    for label in info["label"]:
        label_dir = os.path.join(input_dir, label)

        # Load in train_images
        train_images_dir = os.path.join(label_dir, "train_images")
        for img_name in os.listdir(train_images_dir):
            X_train.append(os.path.join(train_images_dir, img_name))
            y_train.append(info["label_map"][label])

        # Load in test_images
        test_images_dir = os.path.join(label_dir, "test_images")
        for img_name in os.listdir(test_images_dir):
            X_test.append(os.path.join(test_images_dir, img_name))
            y_test.append(info["label_map"][label])

    kf = StratifiedKFold(n_splits=k, random_state=random_state, shuffle=True)
    info["train_splits"] = list(kf.split(X_train, y_train))
    return X_train, y_train, X_test, y_test, info


def load_dataset_from_list(
    image_paths: list[str],
    labels: list,
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    is_test_set: bool = False,
    img_augmentation_layers: list[BaseImagePreprocessingLayer] = None,
):
    def input_preprocess_train(path: str, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)

        # Data augmentation
        if (img_augmentation_layers is not None) and (len(img_augmentation_layers) > 0):
            for layer in img_augmentation_layers:
                img = layer(img)

        return img, tf.one_hot(label, depth=num_classes)

    def input_preprocess_test(path: str, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        return img, tf.one_hot(label, depth=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(
        input_preprocess_test if is_test_set else input_preprocess_train,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size=batch_size)
    if not is_test_set:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
