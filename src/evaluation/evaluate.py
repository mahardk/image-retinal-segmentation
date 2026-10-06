import numpy as np


def confusion_counts(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    tn = np.logical_and(
        ~prediction,
        ~ground_truth,
    ).sum()

    fp = np.logical_and(
        prediction,
        ~ground_truth,
    ).sum()

    fn = np.logical_and(
        ~prediction,
        ground_truth,
    ).sum()

    return (
        int(tp),
        int(tn),
        int(fp),
        int(fn),
    )


def calculate_metrics(
    prediction,
    ground_truth,
):

    tp, tn, fp, fn = confusion_counts(
        prediction,
        ground_truth,
    )

    eps = 1e-8

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn + eps)
    )

    sensitivity = (
        tp
        / (tp + fn + eps)
    )

    specificity = (
        tn
        / (tn + fp + eps)
    )

    f1 = (
        2 * tp
        / (2 * tp + fp + fn + eps)
    )

    iou = (
        tp
        / (tp + fp + fn + eps)
    )

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-Score": f1,
        "IoU": iou,
    }