"""Configurable U-Net adapted from the original class project."""
import tensorflow as tf


def conv_block(inputs, filters, residual=False, activation='relu'):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if residual:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')(inputs)
        x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation(activation)(x)


def attention_gate(skip, gate, filters, activation='relu'):
    theta = tf.keras.layers.Conv2D(filters, 1, padding='same')(skip)
    theta = tf.keras.layers.BatchNormalization()(theta)
    phi = tf.keras.layers.Conv2D(filters, 1, padding='same')(gate)
    phi = tf.keras.layers.BatchNormalization()(phi)
    x = tf.keras.layers.Activation(activation)(tf.keras.layers.Add()([theta, phi]))
    coefficients = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    return tf.keras.layers.Multiply()([skip, coefficients])


def unet_model(input_shape=(128, 128, 1), num_classes=1, attention=False,
               residual=False, activation='relu', base_filters=16):
    """Four encoder/decoder levels; use base_filters=64 for original width."""
    if base_filters < 2 or num_classes != 1:
        raise ValueError('At least two base filters and one binary output are required')
    if len(input_shape) != 3 or any(s is None or s < 16 or s % 16 for s in input_shape[:2]):
        raise ValueError('Spatial dimensions must be positive multiples of 16')
    inputs = tf.keras.Input(shape=input_shape)
    x, skips = inputs, []
    for level in range(4):
        filters = base_filters * 2**level
        x = conv_block(x, filters, residual, activation)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x, base_filters * 16, residual, activation)
    for level, skip in reversed(list(enumerate(skips))):
        filters = base_filters * 2**level
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        if attention:
            skip = attention_gate(skip, x, max(1, filters // 2), activation)
        x = conv_block(tf.keras.layers.Concatenate()([x, skip]), filters, residual, activation)
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='U-Net')


def hard_dice_per_image(y_true, y_pred):
    """Return a vector so Keras averages samples rather than batches."""
    truth = tf.cast(y_true, tf.float32)
    pred = tf.cast(y_pred > .5, tf.float32)
    intersection = tf.reduce_sum(truth * pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(truth + pred, axis=(1, 2, 3))
    return tf.where(denominator > 0, tf.math.divide_no_nan(2 * intersection, denominator), tf.ones_like(denominator))


def dice_loss(y_true, y_pred, smooth=1e-6):
    truth = tf.cast(y_true, y_pred.dtype)
    intersection = tf.reduce_sum(truth * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(truth + y_pred, axis=(1, 2, 3))
    return 1 - tf.reduce_mean((2 * intersection + smooth) / (denominator + smooth))


def iou_loss(y_true, y_pred, smooth=1e-6):
    truth = tf.cast(y_true, y_pred.dtype)
    intersection = tf.reduce_sum(truth * y_pred, axis=(1, 2, 3))
    union = tf.reduce_sum(truth + y_pred, axis=(1, 2, 3)) - intersection
    return 1 - tf.reduce_mean((intersection + smooth) / (union + smooth))
