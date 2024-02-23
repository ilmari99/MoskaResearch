#!/usr/bin/env python3
import os
import sys
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import argparse
from read_to_dataset import read_to_dataset#_old as read_to_dataset
import re
import datetime

def get_base_model(input_shape):
    """ Same model used in the original study
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(600, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    x = tf.keras.layers.Dense(550, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    x = tf.keras.layers.Dense(450, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model")
    # Datafolders is a list of strings, so we need to evaluate it as a list of strings
    #parser.add_argument("data_folders", help="The data folders to train on", default="./FullyRandomDataset-V1/",nargs='?')
    
    # Passed as: --datasets FullyRandomDataset-V1 FullyRandomDataset-V2
    parser.add_argument("--datasets", nargs="*", type=str, help="The datasets to train on", required=True)
    
    # PAssed as: --model_file "ModelH5Files/1110_basic_notop_V6V7V7-1V8_5593.h5"
    parser.add_argument("--pre_trained_model_file", help="Load a pre-trained model to finetune.", default="")
    
    parser.add_argument("--model_out_file", help="The file to save the model to. If not specified, the standard naming convention is used.", default="")
    parser.add_argument("--input_shape", help="The input shape of the model", default="(442,)")
    parser.add_argument("--batch_size", help="The batch size", default="16384")
    parser.add_argument("--validation_split", help="Which fraction to use for validation and test sets.", default="0.05")
    parser.add_argument("--patience", help="The patience for early stopping", default="6")
    
    print(f"Traning model in cwd: {os.getcwd()}")
    print(f"With arguments: {parser.parse_args()}")
    print(f"Tensorflow version: {tf.__version__}")
    print(f"nvcc --version: {os.popen('nvcc --version').read()}")
    print(f"Python version: {sys.version}")
    print('Tensorflow GPUS:', tf.config.list_physical_devices('GPU'))
    
    parser = parser.parse_args()
    
    DATA_FOLDERS = parser.datasets
    #"Vectors" folder
    DATA_FOLDERS = [path + os.sep + "Vectors" for path in DATA_FOLDERS if not path.endswith("Vectors")]
    
    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            raise ValueError(f"Folder {folder} does not exist")
    if not os.path.exists(parser.pre_trained_model_file) and parser.pre_trained_model_file != "":
        raise ValueError(f"Model file {parser.pre_trained_model_file} does not exist")
    if os.path.exists(parser.model_out_file):
        raise ValueError(f"Model file {parser.model_out_file} already exists")
    
    
    INPUT_SHAPE = eval(parser.input_shape)
    BATCH_SIZE = int(parser.batch_size)
    VALIDATION_SPLIT = float(parser.validation_split)
    from_loaded_model = True if parser.pre_trained_model_file != "" else False
    is_conv = False if len(INPUT_SHAPE) == 1 else True
    PATIENCE = int(parser.patience)
    
    strategy = tf.distribute.MirroredStrategy(devices=[g.name.replace("physical_device:", "") for g in tf.config.list_physical_devices('GPU')])
    
    with strategy.scope():

        model = tf.keras.models.load_model(parser.pre_trained_model_file) if from_loaded_model else get_base_model(INPUT_SHAPE)

        ds, n_files = read_to_dataset(DATA_FOLDERS,
            add_channel = True if INPUT_SHAPE[-1] == 1 else False,
            shuffle_files=True,
        )
        print(f"Dataset read {n_files} files")
        
        print(f"Loaded model architecture: {model.summary()}")
        print(f"Model file used: {parser.pre_trained_model_file}" if from_loaded_model else "No model file used")

        approx_num_states = 80 * n_files
        
        VALIDATION_LENGTH = int(VALIDATION_SPLIT  * approx_num_states)
        TEST_LENGTH = int(VALIDATION_SPLIT  * approx_num_states)
        print(f"Total number of samples (approximate): {approx_num_states}")
        print(f"Validation and test length: {VALIDATION_LENGTH} and {TEST_LENGTH}")
        
        SHUFFLE_BUFFER_SIZE = 2*BATCH_SIZE

        # For naming the model file, we need date, dataset versions, conv or not conv, on top of a model or not
        # Find the dataset version number (V1, V2, V3, V7-1, etc)
        reg = re.compile(r"V\d ?-?\d?")
        if len(reg.findall(DATA_FOLDERS[0])) > 0: 
            ds_versions = [reg.findall(folder)[0] for folder in DATA_FOLDERS]
        else:
            ds_versions = ["UNK"]
        #DDMM
        date = datetime.datetime.now().strftime("%d%m")
        # DDMM_<type>_<from_model>_<dataset_versions>.h5
        parts = []
        parts.append(date)
        parts.append("conv" if is_conv else "basic")
        parts.append("top" if from_loaded_model else "notop")
        parts.append("".join(ds_versions))
        default_model_file = "_".join(parts) + ".h5"
        
        model_file = parser.model_out_file if parser.model_out_file != "" else default_model_file
        print("When the model is trained, it will be saved to: ",model_file)

        model_main_name = os.path.splitext(os.path.basename(model_file))[0]
        model_file_path = f"ModelH5Files/{model_file}"
        tensorboard_log = f"Tensorboard-logs/{model_main_name}"
        checkpoint_filepath = f"NNCheckpoints/{model_main_name}"
        
        if os.path.exists(tensorboard_log):
            warnings.warn("Tensorboard log directory already exists!")
        if model_file is None:
            raise ValueError("Model file must be specified")
        
        print("Tensorboard log directory: ",tensorboard_log)
        print("Checkpoint directory: ",checkpoint_filepath)
        print("Model file: ",model_file)
        
        validation_ds = ds.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
        test_ds = ds.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
        train_ds = ds.skip(VALIDATION_LENGTH+TEST_LENGTH).shuffle(SHUFFLE_BUFFER_SIZE)
        train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=PATIENCE, restore_best_weights=True, start_from_epoch=0)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log,
                                                        histogram_freq=3,
                                                        write_steps_per_second=True,
                                                        profile_batch=(100,150)
                                                        )
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        model.fit(x=train_ds,
                  validation_data=validation_ds,
                  epochs=5, 
                  callbacks=[early_stopping_cb, tensorboard_cb, model_checkpoint_callback],
                  #max_queue_size=300,
                  #workers=60,
                  #use_multiprocessing=False,
                  )
        
        result = model.evaluate(test_ds, verbose=2)

        # Add the first 4 decimals of test loss to the model file name
        loss = str(round(result[0],4))
        loss = loss[2:]
        model_file = model_file_path[:-3] + f"_{loss}.h5"

        # Os change the Tensorboard log and checkpoint directory names to include the loss
        os.rename(tensorboard_log, tensorboard_log + "_" + loss)
        os.rename(checkpoint_filepath, checkpoint_filepath + "_" + loss)
        model.save(model_file)
