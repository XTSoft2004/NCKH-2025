import keras
from typing import Callable
from keras import layers, models as kmodels


def final_head(
    x, num_classes: int, dropout_rate: float = 0.5, name_prefix: str = "head"
):
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    x = layers.Dense(256, activation="relu", name=f"{name_prefix}_fc")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name=f"{name_prefix}_out"
    )(x)
    return outputs


def build_mobilenetv3_small(image_size: tuple[int, int], num_classes: int):
    base = keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    x = base(inputs, training=False)
    outputs = final_head(x, num_classes, name_prefix="mnetv3_small")
    return kmodels.Model(inputs, outputs, name="MobileNetV3_Small_Backbone")


def build_mobilenetv3_large(image_size: tuple[int, int], num_classes: int):
    base = keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    x = base(inputs, training=False)
    outputs = final_head(x, num_classes, name_prefix="mnetv3_large")
    return kmodels.Model(inputs, outputs, name="MobileNetV3_Large_Backbone")


def build_mobilenetv2(image_size: tuple[int, int], num_classes: int, alpha=1.0):
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
        alpha=alpha,
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    x = base(inputs, training=False)
    outputs = final_head(x, num_classes, name_prefix="mnetv2")
    return kmodels.Model(inputs, outputs, name=f"MobileNetV2_Alpha-{alpha}_Backbone")


def build_mobilenet(image_size: tuple[int, int], num_classes: int, alpha=1.0):
    base = keras.applications.MobileNet(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
        alpha=alpha,
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    x = base(inputs, training=False)
    outputs = final_head(x, num_classes, name_prefix="mnet")
    return kmodels.Model(inputs, outputs, name=f"MobileNet_Alpha-{alpha}_Backbone")


models_dict: dict[str, Callable] = {
    "mnetv3_small": build_mobilenetv3_small,
    "mnetv3_large": build_mobilenetv3_large,
    "mnetv2": build_mobilenetv2,
    "mnet": build_mobilenet,
}
