
# tensorflow model
def create_tf_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2000, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

