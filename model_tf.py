
import tensorflow as tf
# Define the TensorFlow model
def create_tf_model():
    tf.random.set_seed(89)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(2313,),
                              kernel_regularizer=tf.keras.regularizers.l2(0.0069)),
        tf.keras.layers.Dropout(0.37, seed=12),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0069)),
        tf.keras.layers.Dropout(0.37, seed=12),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0069)),
        tf.keras.layers.Dropout(0.37, seed=12),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Create the optimizer with the desired learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00126)

    # Compile the model with the optimizer
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

