#!/usr/bin/env python3
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def create_tf_dataset(paths, add_channel=False,get_part="full") -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        try:
            paths = [paths]
        except:
            raise ValueError("Paths should must be a list of strings")
    # Cards on table, vards in hand + players ready, players out, kopled
    misc_parts = list(range(0,5)) + list(range(161,171))
    card_parts = list((n for n in range(0,431) if n not in misc_parts))
    print(f"Number of card parts" + str(len(card_parts)))
    print(f"Misc parts: {misc_parts}")
    file_paths = []
    for path in paths:
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a directory")
        file_paths += [os.path.join(path, file) for file in os.listdir(path)]
    print("Number of files: " + str(len(file_paths)))
    random.shuffle(file_paths)
    print("Shuffled files.")
    dataset = tf.data.TextLineDataset(file_paths)
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=", "), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1]), tf.strings.to_number(x[-1])), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print(f"Getting part: {get_part}")
    # Get only the parts we want
    if get_part == "cards":
        dataset = dataset.map(lambda x,y: (tf.gather(x, card_parts), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif get_part == "misc":
        dataset = dataset.map(lambda x,y: (tf.gather(x, misc_parts), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif get_part != "full":
        raise ValueError(f"get_part should be 'cards', 'misc' or 'full', not {get_part}")
    
    # Add a channel dimension
    if add_channel:
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), y))
    return dataset


def get_loaded_model(path) -> tf.keras.models.Sequential:
    model = tf.keras.models.load_model(path,compile=True)
    return model
    

def load_from_checkpoint(model : tf.keras.models.Sequential, checkpoint_path : str) -> tf.keras.models.Sequential:
    model.load_weights(checkpoint_path,)
    return model
    
def get_transfer_model(base_model_path):
    global INPUT_SHAPE
    base_model = tf.keras.models.load_model(base_model_path,compile=True)
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(431, activation="linear")(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(425, activation="linear")(x)
    output = base_model(x, training=False)
    new_model = tf.keras.Model(inputs, output)
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        metrics = ["accuracy"],
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0)
    )
    return new_model

def get_nn_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.Dense(600, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.35))
    model.add(tf.keras.layers.Dense(550, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.35))
    model.add(tf.keras.layers.Dense(500, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(450, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015, amsgrad=False),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy']
        )
    return model

def get_card_model(standalone = False, compile_ = True):
    model = tf.keras.models.Sequential()
    if standalone:
        model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.Conv1D(128,3,activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(32,12, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Flatten())
    if standalone:
        model.add(tf.keras.layers.Dense(400,activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.4))
        model.add(tf.keras.layers.Dense(300,activation="relu"))
        model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    if compile_:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015, amsgrad=False),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
            metrics=['accuracy']
        )
    return model

def get_misc_model(compile_=True):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(12,activation="relu"))
    #model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(12,activation="relu"))
    #model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(12,activation="relu"))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    if compile_:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015, amsgrad=False),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
            metrics=['accuracy']
        )
    return model


def get_branched_model():
    misc_parts = list(range(0,5)) + list(range(161,171))
    card_parts = list((n for n in range(0,431) if n not in misc_parts))
    
    inputs = tf.keras.layers.Input(shape=(430,))
    card_data = tf.gather(inputs, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430])
    card_data = tf.expand_dims(card_data, axis=-1)
    misc_data = tf.gather(inputs, [0, 1, 2, 3, 4, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170])
    
    # Model for card input (416,1); its like an image
    img = tf.keras.layers.Conv1D(128,3,activation="linear")(card_data)
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Conv1D(32,12, activation="linear")(img)
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Flatten()(img)
    img = tf.keras.layers.Dense(400,activation="relu")(img)
    img = tf.keras.layers.Dropout(rate=0.4)(img)
    img = tf.keras.layers.Dense(300,activation="relu")(img)
    img_out = tf.keras.layers.Dense(1,activation="sigmoid")(img)
    
    # Model for misc input (14,)
    misc = tf.keras.layers.Dense(15,activation="relu")(misc_data)
    misc = tf.keras.layers.Dropout(rate=0.3)(misc)
    misc = tf.keras.layers.Dense(12,activation="relu")(misc)
    misc = tf.keras.layers.Dropout(rate=0.3)(misc)
    misc = tf.keras.layers.Dense(12,activation="relu")(misc)
    misc_out = tf.keras.layers.Dense(1,activation="sigmoid")(misc)
    
    final_out = tf.keras.layers.concatenate()[tf.flatten(img_out),misc_out]
    final_out = tf.keras.layers.Dense(2,activation="linear")(final_out)
    final_out = tf.keras.layers.Dense(1,activation="sigmoid")(final_out)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=final_out, name="branched_model")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015, amsgrad=False),
        loss=[tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0)],
        metrics=['accuracy'])
    return model
    
    
    

def get_conv_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.Conv1D(32,3,activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
    model.add(tf.keras.layers.Conv1D(32,6, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(32,12, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(500,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Dense(400,activation="relu"))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2,from_logits=False,label_smoothing=0),
        metrics=['accuracy']
    )
    return model

INPUT_SHAPE = (430,1)
if __name__ == "__main__":
    
    all_dataset = create_tf_dataset(["./Data/LastLogs/Vectors/"],
                                    add_channel=True,
                                    get_part="full"
                                    )
    print(all_dataset.take(1).as_numpy_iterator().next())
    #model = load_from_checkpoint(get_nn_model(),'./model-checkpoints/')
    model = get_conv_model()
    VALIDATION_LENGTH = 100000
    TEST_LENGTH = 100000
    BATCH_SIZE = 4096
    tensorboard_log = "tensorboard-log/"
    checkpoint_filepath = './model-checkpoints/'
    model_file = "model.h5"    

    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)#Add shuffle for NN
    
    if os.path.exists(tensorboard_log):
        raise Exception("Tensorboard log directory already exists")
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=20, restore_best_weights=True, start_from_epoch=10)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log,histogram_freq=5,profile_batch=(50,100),)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    model.fit(x=train_ds, 
              validation_data=validation_ds, 
              epochs=100, 
              callbacks=[early_stopping_cb, tensorboard_cb, model_checkpoint_callback],
              )
    
    model.evaluate(test_ds, verbose=2)
    
    model.save(model_file)
