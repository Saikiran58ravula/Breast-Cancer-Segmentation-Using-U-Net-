"""
U-Net model architecture and custom loss/metric functions
for breast cancer ultrasound image segmentation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient metric — measures overlap between predicted and true masks."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def focal_loss(y_true, y_pred, gamma=2):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    pt = tf.exp(-bce)
    return K.mean((1 - pt) ** gamma * bce)


def combined_loss(y_true, y_pred):
    """Combined Dice + Focal loss — handles class imbalance better than either alone."""
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def build_unet(input_shape=(256, 256, 1)):
    """Builds the U-Net model: encoder -> bottleneck -> decoder with skip connections."""
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    c4 = layers.Dropout(0.4)(c4)

    u5 = layers.UpSampling2D()(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = conv_block(u5, 128)

    u6 = layers.UpSampling2D()(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = conv_block(u6, 64)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = conv_block(u7, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    model = models.Model(inputs, outputs)
    return model


def get_custom_objects():
    """Custom objects needed when loading a saved model (predict.py / evaluate.py)."""
    return {
        "dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "focal_loss": focal_loss,
        "combined_loss": combined_loss,
    }
