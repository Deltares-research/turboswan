import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Cropping2D,
    Dropout,
    UpSampling2D,
    ZeroPadding2D,
)


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
        height, width = self.input_shape[0], self.input_shape[1]
        divisor = 2 ** (len(self.widths) - 2)  # one halving per DownBlock stage
        self._height_padding = self._axis_padding(height, divisor)
        self._width_padding = self._axis_padding(width, divisor)
        x = ZeroPadding2D((self._height_padding, self._width_padding))(model_input)
        x = self.ResidualBlock(self.widths[0])(x)
        return x

    def create_output_model(self, x):
        x = self.ResidualBlock(self.widths[0])(x)
        x = Cropping2D((self._height_padding, self._width_padding))(x)
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
    def _axis_padding(size: int, divisor: int) -> tuple:
        """Pad size up to the next multiple of divisor, split as evenly as possible."""
        remainder = size % divisor
        total_pad = 0 if remainder == 0 else divisor - remainder
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        return pad_before, pad_after

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
