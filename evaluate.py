"""
Evaluate the trained U-Net model on the validation set and report
Dice Coefficient, IoU, Accuracy, Precision, and Recall.

Usage:
    python evaluate.py --model_path models/best_model.h5 --dataset_path dataset/Dataset_BUSI_with_GT
"""

import argparse
import numpy as np
import tensorflow as tf

from utils.data_loader import build_datasets
from utils.model import get_custom_objects


def compute_metrics(y_true, y_pred, threshold=0.4, smooth=1e-7):
    y_true = y_true.flatten()
    y_pred = (y_pred.flatten() > threshold).astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    dice = (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    iou = (intersection + smooth) / (union + smooth)

    tp = intersection
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_true) - tp
    tn = len(y_true) - tp - fp - fn

    accuracy = (tp + tn) / len(y_true)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return {
        "Dice Coefficient": dice,
        "IoU": iou,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }


def main(args):
    model = tf.keras.models.load_model(
        args.model_path, custom_objects=get_custom_objects()
    )

    _, _, (X_val, y_val) = build_datasets(dataset_path=args.dataset_path)

    preds = model.predict(X_val, batch_size=args.batch_size)

    metrics = compute_metrics(y_val, preds, threshold=args.threshold)

    print("\n--- Evaluation Results ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    if args.save_path:
        with open(args.save_path, "w") as f:
            f.write("# Model Performance\n\n")
            f.write("| Metric | Score |\n|---|---|\n")
            for name, value in metrics.items():
                f.write(f"| {name} | {value:.4f} |\n")
        print(f"\nResults saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net segmentation model")
    parser.add_argument("--model_path", type=str, default="models/best_model.h5")
    parser.add_argument("--dataset_path", type=str,
                         default="dataset/Dataset_BUSI_with_GT")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--save_path", type=str, default=None,
                         help="Optional path to save results as a markdown table")
    args = parser.parse_args()

    main(args)
