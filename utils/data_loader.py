"""
Dataset loading, preprocessing, and tf.data pipeline construction
for the Breast Ultrasound (BUSI) dataset.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

IMG_SIZE = 256
BATCH_SIZE = 8


def load_images_and_masks(dataset_path):
    """
    Loads grayscale ultrasound images and their corresponding masks.

    Expects the BUSI dataset structure:
        dataset_path/
            benign/    image.png, image_mask.png, ...
            malignant/ image.png, image_mask.png, ...
            normal/    image.png, image_mask.png, ...
    """
    images = []
    masks = []

    for folder in ["benign", "malignant", "normal"]:
        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            print(f"Warning: folder not found, skipping: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".png") and "_mask" not in file:
                img_path = os.path.join(folder_path, file)
                mask_path = os.path.join(folder_path, file.replace(".png", "_mask.png"))

                if os.path.exists(mask_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

                    images.append(img)
                    masks.append(mask)

    X = np.array(images)
    y = np.array(masks)

    print("Loaded:", X.shape, y.shape)
    return X, y


def preprocess(X, y):
    """Normalize images to [0,1], binarize masks, add channel dimension."""
    X = X.astype("float32") / 255.0
    y = (y > 0).astype("float32")

    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)

    return X, y


def get_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])


def build_datasets(dataset_path, test_size=0.1, random_state=42, batch_size=BATCH_SIZE):
    """
    Loads the dataset and returns ready-to-use train/val tf.data.Dataset objects,
    plus the raw validation arrays (useful for evaluate.py / predict.py).
    """
    X, y = load_images_and_masks(dataset_path)
    X, y = preprocess(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)

    data_augmentation = get_data_augmentation()

    def augment(image, mask):
        image = data_augmentation(image)
        return image, mask

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(augment)
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset, (X_val, y_val)
