"""
Run the trained U-Net model on validation images and visualize predictions
(input image | ground truth mask | predicted mask).

Usage:
    python predict.py --model_path models/best_model.h5 --dataset_path dataset/Dataset_BUSI_with_GT
"""

import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.data_loader import build_datasets
from utils.model import get_custom_objects


def show_predictions(model, dataset, num=3, threshold=0.4, save_dir=None):
    for images, masks in dataset.take(1):
        preds = model.predict(images)

        for i in range(num):
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(images[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(masks[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(preds[i].squeeze() > threshold, cmap="gray")
            plt.axis("off")

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"prediction_sample_{i + 1}.png")
                plt.savefig(out_path)
                print(f"Saved: {out_path}")
            else:
                plt.show()

            plt.close()


def main(args):
    model = tf.keras.models.load_model(
        args.model_path, custom_objects=get_custom_objects()
    )

    _, val_dataset, _ = build_datasets(dataset_path=args.dataset_path)

    show_predictions(
        model, val_dataset,
        num=args.num_samples,
        threshold=args.threshold,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize U-Net predictions")
    parser.add_argument("--model_path", type=str, default="models/best_model.h5")
    parser.add_argument("--dataset_path", type=str,
                         default="dataset/Dataset_BUSI_with_GT")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.4,
                         help="Probability threshold for converting predicted mask to binary")
    parser.add_argument("--save_dir", type=str, default="screenshots",
                         help="Directory to save prediction images (omit to display instead)")
    args = parser.parse_args()

    main(args)
