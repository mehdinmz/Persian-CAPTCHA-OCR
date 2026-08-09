import tensorflow as tf

def get_dataset(path):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.3,
        subset="training",
        seed=225,
        image_size=(28,28),
        batch_size=8,
        color_mode="grayscale"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.3,
        subset="validation",
        seed=225,
        image_size=(28,28),
        batch_size=8,
        color_mode="grayscale"
    )

    rescale = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(
        lambda x, y: (rescale(x), y)
    )

    val_ds = val_ds.map(
        lambda x, y: (rescale(x), y)
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds