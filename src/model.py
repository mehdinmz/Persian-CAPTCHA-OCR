import tensorflow as tf


def build_model(
    input_shape=(28, 28, 1),
    num_classes=10
):
    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=input_shape),

        # Block 1
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 2
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classifier
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model