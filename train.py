"""
Train the U-Net model for breast cancer ultrasound segmentation.

Usage:
    python train.py --dataset_path dataset/Dataset_BUSI_with_GT --epochs 40
"""

import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.data_loader import build_datasets
from utils.model import build_unet, combined_loss, dice_coef


def main(args):
    train_dataset, val_dataset, _ = build_datasets(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
    )

    model = build_unet()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=combined_loss,
        metrics=["accuracy", dice_coef],
    )

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.3),
        tf.keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save training curves (Dice score + accuracy) for the README screenshot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["dice_coef"])
    plt.plot(history.history["val_dice_coef"])
    plt.title("Dice Score")
    plt.legend(["Train", "Val"])

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy")
    plt.legend(["Train", "Val"])

    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    plt.savefig(args.metrics_out)
    print(f"Training curves saved to: {args.metrics_out}")
    print(f"Best model saved to: {args.model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for breast cancer segmentation")
    parser.add_argument("--dataset_path", type=str,
                         default="dataset/Dataset_BUSI_with_GT",
                         help="Path to the BUSI dataset (benign/malignant/normal folders)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model_out", type=str, default="models/best_model.h5")
    parser.add_argument("--metrics_out", type=str, default="screenshots/training_metrics.png")
    args = parser.parse_args()

    main(args)
