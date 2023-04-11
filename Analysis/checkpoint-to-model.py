import tensorflow as tf

""" Load a checkpoint and save it as a model. """

def restore_model_from_checkpoint(model, checkpoint_folder):
    checkpoint = tf.train.Checkpoint(model=model)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_folder)
    checkpoint.restore(latest_checkpoint)
    return model

def pre_model():
    def get_nn_model():
        global INPUT_SHAPE
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(442,)))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(tf.keras.layers.Dense(600, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(550, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(500, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(450, activation="relu"))
        #model.add(tf.keras.layers.Dropout(0.4))
        #model.add(tf.keras.layers.Dense(550, activation="relu"))
        #model.add(tf.keras.layers.Dropout(0.3))
        #model.add(tf.keras.layers.Dense(550, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=False),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
            metrics=['accuracy']
            )
        return model
        return model
    return get_nn_model()

if __name__ == "__main__":
    model = pre_model()
    checkpoint_folder = "./Models/Model-nn1-fuller/model-checkpoints/checkpoint"
    save_folder = "./Models/Model-nn1-fuller/restored.h5"
    model = restore_model_from_checkpoint(model, checkpoint_folder)
    model.save(save_folder)