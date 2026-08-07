import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    ZeroPadding2D,
    Cropping2D,
    AveragePooling2D,
    Lambda,
)
import numpy as np
import tensorflow.keras.backend as K


class TurboSwanModel:
    def __init__(
        self,
        input_shape,
        bnd_input_shape,
        output_variables,
        widths,
        k,
        block_depth,
        min_max_values,
        dropout=True,
    ):
        self.input_shape = input_shape
        self.bnd_input_shape = bnd_input_shape
        self.output_variables = output_variables
        self.widths = widths
        self.k = k
        self.block_depth = block_depth
        self.min_max_values = min_max_values
        self.dropout = dropout
        self.model = self.build_model()

    def ResidualBlock(self, width):
        def apply(x):
            input_width = x.shape[-1]
            residual = x if input_width == width else Conv2D(width, kernel_size=1)(x)
            x = BatchNormalization()(x)
            x = Conv2D(width, kernel_size=self.k, padding="same")(x)
            x = BatchNormalization()(x)
            x = Conv2D(width, kernel_size=self.k, padding="same")(x)
            x = tf.keras.layers.Add()([x, residual])
            if self.dropout:
                x = Dropout(0.2)(x)
            return x

        return apply

    def ResidualBlockTranspose(self, width):
        def apply(x):
            input_width = x.shape[-1]
            residual = x if input_width == width else Conv2D(width, kernel_size=1)(x)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(width, kernel_size=self.k, padding="same")(x)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(width, kernel_size=self.k, padding="same")(x)
            x = tf.keras.layers.Add()([x, residual])
            if self.dropout:
                x = Dropout(0.2)(x)
            return x

        return apply

    def DownBlock(self, width):
        def apply(inputs):
            x, skips = inputs
            for _ in range(self.block_depth):
                x = self.ResidualBlock(width)(x)
                skips.append(x)
            x = AveragePooling2D(pool_size=2)(x)
            return x

        return apply

    def UpBlock(self, width):
        def apply(inputs):
            x, skips = inputs
            x = UpSampling2D(size=2)(x)
            for _ in range(self.block_depth):
                x = Concatenate()([x, skips.pop()])
                x = self.ResidualBlockTranspose(width)(x)
            return x

        return apply

    def create_input_model(self, model_input):
        # x = Lambda(self.conditional_padding_to_512)(model_input)
        x = ZeroPadding2D(((15, 16), (45, 46)))(model_input)
        x = self.ResidualBlock(self.widths[0])(x)
        return x

    def create_output_model(self, x):
        x = self.ResidualBlock(self.widths[0])(x)
        # x = Lambda(lambda t: self.conditional_crop_to_original(t))(x)
        x = Cropping2D(((15, 16), (45, 46)))(x)
        x = Conv2DTranspose(
            len(self.output_variables),
            kernel_size=1,
            padding="valid",
            activation="linear",
            name="output",
        )(x)
        return x

    def build_model(self):
        img_input = Input(self.input_shape)
        x = self.create_input_model(img_input)

        skips = []
        for width in self.widths[1:-1]:
            x = self.DownBlock(width)([x, skips])

        for _ in range(self.block_depth):
            x = self.ResidualBlock(self.widths[-1])(x)

        img_shape = K.int_shape(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        bnd_input = Input(self.bnd_input_shape)
        bnd_x = tf.keras.layers.Flatten()(bnd_input)
        bnd_x = tf.keras.layers.Dense(
            self.bnd_input_shape[0] * self.bnd_input_shape[1]
        )(bnd_x)
        bnd_x = tf.keras.layers.BatchNormalization()(bnd_x)
        bnd_x = tf.keras.layers.Dense(128)(bnd_x)
        bnd_x = tf.keras.layers.BatchNormalization()(bnd_x)
        bnd_x = tf.keras.layers.Dense(256)(bnd_x)
        bnd_x = tf.keras.layers.BatchNormalization()(bnd_x)
        bnd_x = tf.keras.layers.Dense(256)(bnd_x)
        bnd_x = tf.keras.layers.BatchNormalization()(bnd_x)

        x = tf.keras.layers.Concatenate()([x, bnd_x])
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(img_shape[1] * img_shape[2] * img_shape[3])(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Reshape((img_shape[1], img_shape[2], img_shape[3]))(x)

        for width in reversed(self.widths[1:-1]):
            x = self.UpBlock(width)([x, skips])

        output = self.create_output_model(x)

        return Model(inputs=[img_input, bnd_input], outputs=[output])

    def denormalize(self, x, var):
        minimum, maximum = (
            self.min_max_values[var].values[0],
            self.min_max_values[var].values[1],
        )
        return (x * (maximum - minimum)) + minimum

    @staticmethod
    def conditional_padding_to_512(x):
        input_shape = tf.shape(x)
        height_pad = tf.maximum(512 - input_shape[1], 0)
        width_pad = tf.maximum(512 - input_shape[2], 0)

        pad_top = height_pad // 2
        pad_bottom = height_pad - pad_top
        pad_left = width_pad // 2
        pad_right = width_pad - pad_left

        paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
        return tf.cond(
            tf.logical_or(height_pad > 0, width_pad > 0),
            lambda: tf.pad(x, paddings),
            lambda: x,
        )

    @staticmethod
    def conditional_crop_to_original(x, target_height=481, target_width=421):
        input_shape = tf.shape(x)
        height_crop = input_shape[1] - target_height
        width_crop = input_shape[2] - target_width

        crop_top = height_crop // 2
        crop_bottom = height_crop - crop_top
        crop_left = width_crop // 2
        crop_right = width_crop - crop_left

        cropped = x[
            :,
            crop_top : input_shape[1] - crop_bottom,
            crop_left : input_shape[2] - crop_right,
            :,
        ]
        return tf.cond(
            tf.logical_or(height_crop > 0, width_crop > 0), lambda: cropped, lambda: x
        )

    @staticmethod
    def circular_error(a, b, to_rad=True):
        if to_rad:
            a = a * (2 * np.pi) / 360
            b = b * (2 * np.pi) / 360
        a = a % (2 * np.pi)
        b = b % (2 * np.pi)
        diff = tf.math.abs(a - b)
        return tf.math.minimum(diff, 2 * np.pi - diff) / (2 * np.pi)

    @staticmethod
    def masked_loss(mask=None):  # `mask` is unused here, but kept for compatibility
        def loss(y_true, y_pred):
            # Assume y_true shape: (batch, H, W, C+1), where last channel is mask
            mask = y_true[..., -1]  # shape: (batch, H, W)
            y_true_val = y_true[..., :-1]  # shape: (batch, H, W, C)

            # Flatten everything to apply mask
            mask_flat = tf.reshape(mask, [-1])
            y_true_flat = tf.reshape(y_true_val, [-1, tf.shape(y_true_val)[-1]])
            y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

            # Apply mask
            masked_true = tf.boolean_mask(y_true_flat, mask_flat)
            masked_pred = tf.boolean_mask(y_pred_flat, mask_flat)

            return tf.reduce_mean(tf.square(masked_true - masked_pred))

        return loss
