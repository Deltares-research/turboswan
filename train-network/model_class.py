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
)
import numpy as np
import tensorflow.keras.backend as K


class TurboSwanModel:
    def __init__(
        self,
        input_shape,
        output_variables,
        widths,
        k,
        block_depth,
        min_max_values,
        dropout=True,
    ):
        self.input_shape = input_shape
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
            x = MaxPooling2D(pool_size=2)(x)
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
        x = ZeroPadding2D(((15, 16), (45, 46)))(model_input)
        x = self.ResidualBlock(self.widths[0])(x)
        return x

    def create_output_model(self, x):
        x = self.ResidualBlock(self.widths[0])(x)
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

        for width in reversed(self.widths[1:-1]):
            x = self.UpBlock(width)([x, skips])

        output = self.create_output_model(x)
        return Model(inputs=[img_input], outputs=[output])

    def denormalize(self, x, var):
        minimum, maximum = (
            self.min_max_values[var].values[0],
            self.min_max_values[var].values[1],
        )
        return (x * (maximum - minimum)) + minimum

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
    def masked_loss():
        def loss(y_true, y_pred):
            mask = y_true[..., -1]
            true_values = y_true[..., :-1]
            true_values = tf.expand_dims(true_values, axis=-1)
            mask_expanded = tf.expand_dims(mask, axis=-1)
            # mask_tiled = tf.tile(mask_expanded, [tf.shape(true_values)[0], 1, 1])
            masked_true = tf.boolean_mask(true_values, mask_expanded)
            masked_pred = tf.boolean_mask(y_pred, mask_expanded)
            return K.mean(tf.square(masked_true - masked_pred))

        return loss
