import keras
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc


def evaluate(ds: tf.data.Dataset, model: keras.Model, class_map: dict[str, int]):
    preds: np.ndarray = model.predict(ds)
    y_pred: list[int] = preds.argmax(axis=1)

    y_true_onehot = []
    for _, labels in ds:  # type: ignore
        y_true_onehot.append(labels.numpy())
    y_true_onehot = np.concatenate(y_true_onehot, axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)

    print(classification_report(y_true, y_pred, target_names=class_map.keys()))  # type: ignore
    result = classification_report(
        y_true, y_pred, target_names=class_map.keys(), output_dict=True  # type: ignore
    )

    y_pred_proba = preds[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    result["auc"] = auc(fpr, tpr)  # type: ignore
    result["y_true"] = y_true  # type: ignore
    result["y_pred_proba"] = y_pred_proba  # type: ignore

    return result
