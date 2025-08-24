from typing import Callable
import keras
from keras import layers, models as kmodels

data_augmentation = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
    ]
)


def final_head(
    x, num_classes: int, dropout_rate: float = 0.5, name_prefix: str = "head"
):
    """
    Builds the final classification head for a neural network model.

    Args:
      x: Input tensor to the head.
      num_classes (int): Number of output classes for classification.
      dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.5.
      name_prefix (str, optional): Prefix for layer names. Defaults to "head".

    Returns:
      Tensor: Output tensor with shape (num_classes,) and softmax activation.
    """
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    x = layers.Dense(128, activation="relu", name=f"{name_prefix}_fc")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name=f"{name_prefix}_out"
    )(x)
    return outputs


def build_mobilenetv3_small(image_size: tuple[int, int], num_classes: int):
    """
    Builds a custom MobileNetV3 Small model for image classification.

    Args:
        image_size (tuple): The input image size as (height, width).
        num_classes (int): Number of output classes for classification.

    Returns:
        keras.Model: A Keras model instance with MobileNetV3 Small as the base and a custom classification head.

    Notes:
        - The base MobileNetV3 Small model is loaded with ImageNet weights and is not trainable.
        - The final classification head is added with a dropout rate of 0.5.
    """
    base = keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
        pooling=None,
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    inputs = data_augmentation(inputs)
    x = base(inputs, training=False)
    outputs = final_head(x, num_classes, dropout_rate=0.5, name_prefix="mnetv3_small")
    return kmodels.Model(inputs, outputs, name="MobileNetV3_Small_custom")


def build_mobilenetv3_large(image_size: tuple[int, int], num_classes: int):
    """
    Builds a custom MobileNetV3 Large model for image classification.

    Args:
        image_size (tuple): The input image size as (height, width).
        num_classes (int): The number of output classes for classification.

    Returns:
        keras.Model: A Keras model instance with MobileNetV3 Large as the base and a custom classification head.

    Notes:
        - The base MobileNetV3 Large model is loaded with ImageNet weights and is frozen (not trainable).
        - The model does not include the top classification layer from MobileNetV3 Large.
        - A custom head is added for classification, with dropout applied.
    """
    base = keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
        pooling=None,
        include_preprocessing=False,
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    inputs = data_augmentation(inputs)
    x = base(inputs, training=False)
    outputs = final_head(x, num_classes, dropout_rate=0.5, name_prefix="mnetv3_large")
    return kmodels.Model(inputs, outputs, name="MobileNetV3_Large_custom")


def build_mobilenetv2(image_size: tuple[int, int], num_classes: int, alpha=1.0):
    """
    Builds a MobileNetV2-based image classification model with a custom head.

    Args:
        image_size (tuple): The input image size as (height, width).
        num_classes (int): Number of output classes for classification.
        alpha (float, optional): Width multiplier for the MobileNetV2 model. Defaults to 1.0.

    Returns:
        keras.Model: A Keras Model instance with MobileNetV2 base and custom classification head.
    """
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
        alpha=alpha,
        pooling=None,
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    inputs = data_augmentation(inputs)
    x = base(inputs, training=False)
    outputs = final_head(x, num_classes, dropout_rate=0.5, name_prefix="mnetv2")
    return kmodels.Model(inputs, outputs, name=f"MobileNetV2_alpha{alpha}_custom")


def build_mobilenet_v1(image_size: tuple[int, int], num_classes: int, alpha=1.0):
    base = keras.applications.MobileNet(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
        alpha=alpha,
        pooling=None,
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    inputs = data_augmentation(inputs)
    x = base(inputs, training=False)
    outputs = final_head(x, num_classes, dropout_rate=0.5, name_prefix="mnet_v1")
    return kmodels.Model(inputs, outputs, name=f"MobileNet_v1_alpha{alpha}_custom")


models_dict: dict[str, Callable] = {
    "mnetv3_small": build_mobilenetv3_small,
    "mnetv3_large": build_mobilenetv3_large,
    "mnetv2": build_mobilenetv2,
    "mnet_v1": build_mobilenet_v1,
}
