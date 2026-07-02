# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_accuracy(eval_pred, bos=True):
    outputs, labels = eval_pred

    bs = len(outputs)
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.numpy()
        labels = labels.numpy()
    if bos:
        labels = labels[:, 1:]
        outputs = outputs[:, 1:]
    outputs = np.array(outputs).reshape(bs, -1)
    labels = np.array(labels).reshape(bs, -1)
    correct_predictions = (outputs == labels).astype(int)
    # Calculate accuracy where all positions are considered
    accuracy = correct_predictions.all(axis=-1).mean()
    # Calculate partial accuracy considering the total number of elements
    partial_accuracy = correct_predictions.sum() / np.prod(labels.shape)

    return dict(accuracy=accuracy, partial_accuracy=partial_accuracy)


def compute_accuracy_with_mask(eval_pred, hue=None):
    outputs, labels = eval_pred
    if isinstance(outputs, tuple):
        outputs, masks = outputs
    else:
        masks = None

    if masks is None:
        return compute_accuracy((outputs, labels), bos=False)
    bs = len(outputs)
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.numpy()
        labels = labels.numpy()
        masks = masks.numpy()

    outputs = np.array(outputs).reshape(bs, -1)
    labels = np.array(labels).reshape(bs, -1)
    masks = np.array(masks).reshape(bs, -1)

    correct_predictions = (outputs == labels).astype(int) * masks

    accuracy = (correct_predictions == masks).all(-1).mean()
    partial_accuracy = (
        correct_predictions.sum() / masks.sum() if masks.sum() else np.nan
    )
    return dict(accuracy=accuracy, partial_accuracy=partial_accuracy)


def compute_classification_metrics(eval_pred):
    outputs, labels = eval_pred
    cm = confusion_matrix(
        labels.squeeze(), outputs.squeeze(), labels=[0, 1]
    )  # labels ensure order in confusion matrix
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]
    TP = cm[1, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    recall_label_1 = TP / (TP + FN)
    # Calculate recall for label 0
    recall_label_0 = TN / (TN + FP)
    return dict(
        accuracy=(TP + TN) / cm.sum(), recall_1=recall_label_1, recall_0=recall_label_0
    )

def compute_classification_metrics_acc_only(eval_pred):
    outputs, labels = eval_pred
    cm = confusion_matrix(
        labels.squeeze(), outputs.squeeze(), labels=[0, 1]
    )  # labels ensure order in confusion matrix
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]
    TP = cm[1, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    recall_label_1 = TP / (TP + FN)
    # Calculate recall for label 0
    recall_label_0 = TN / (TN + FP)
    return dict(
        accuracy=(TP + TN) / cm.sum()
    )


def compute_classification_metrics_per_cat_acc_only(eval_pred, hue=None):

    if hue is None:
        return compute_classification_metrics_acc_only(eval_pred)

    outputs, labels = eval_pred
    outputs = outputs.squeeze().numpy()
    labels = labels.squeeze().numpy()

    metrics_by_category = {}

    unique_categories = set(hue)

    for category in unique_categories:
        category_indices = [i for i, cat in enumerate(hue) if cat == category]
        category_outputs = np.array([outputs[i] for i in category_indices])
        category_labels = np.array([labels[i] for i in category_indices])

        cat_metrics = compute_classification_metrics_acc_only(
            (category_outputs, category_labels)
        )

        cat_metrics = {f"{category}/{key}": value for key, value in cat_metrics.items()}
        # Store the metrics
        metrics_by_category.update(cat_metrics)

    metrics_by_category.update(compute_classification_metrics_acc_only(eval_pred))

    return metrics_by_category

def compute_classification_metrics_per_cat(eval_pred, hue=None):

    if hue is None:
        return compute_classification_metrics(eval_pred)

    outputs, labels = eval_pred
    outputs = outputs.squeeze().numpy()
    labels = labels.squeeze().numpy()

    metrics_by_category = {}

    unique_categories = set(hue)

    for category in unique_categories:
        category_indices = [i for i, cat in enumerate(hue) if cat == category]
        category_outputs = np.array([outputs[i] for i in category_indices])
        category_labels = np.array([labels[i] for i in category_indices])

        cat_metrics = compute_classification_metrics(
            (category_outputs, category_labels)
        )

        cat_metrics = {f"{category}/{key}": value for key, value in cat_metrics.items()}
        # Store the metrics
        metrics_by_category.update(cat_metrics)

    metrics_by_category.update(compute_classification_metrics(eval_pred))

    return metrics_by_category
