""" Rank models based on their perforamance on a test dataset.
"""
import os
import argparse
import tensorflow as tf
import random

def read_to_dataset(paths, add_channel = False, shuffle_files = True, return_n_files = False) -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    file_paths = [os.path.join(path, file) for path in paths for file in os.listdir(path) if os.path.isdir(path)]
    if shuffle_files:
        random.shuffle(file_paths)

    print("Found {} files".format(len(file_paths)))

    dataset = tf.data.TextLineDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    # Split all lines by commas to get a list of strings
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=","), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Convert the list of strings to a list ints and take the last element as the label (0,1)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1], out_type=tf.int32), tf.strings.to_number(x[-1], out_type=tf.int32)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Add a channel dimension if necessary
    if add_channel:
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if return_n_files:
        return dataset, len(file_paths)
    return dataset


def get_all_models(folder):
    """Return a list of h5 files in the folder.
    """
    h5_files = []
    for file in os.listdir(folder):
        if file.endswith(".h5") and "conv" not in file:
            h5_files.append(file)
    return h5_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank models based on their performance on a test dataset.")
    parser.add_argument("folder", help="Folder containing the models")
    parser.add_argument("dataset", help="Folder containing the test dataset")
    parser.add_argument("--batch_size", help="Batch size", default=4096)
    
    parser = parser.parse_args()
    
    h5_files = get_all_models(parser.folder)
    ds = read_to_dataset(parser.dataset, add_channel=False)
    ds = ds.batch(int(parser.batch_size))
    model_results = {}
    for h5_file in h5_files:
        model = tf.keras.models.load_model(parser.folder + os.sep + h5_file)
        loss = model.evaluate(ds, batch_size=int(parser.batch_size), verbose=0, return_dict=True)["loss"]
        model_results[h5_file] = loss
        print(f"Model {h5_file} loss: {loss}")
    
    # Show a ranking of the models
    print("Ranking:")
    for model, loss in sorted(model_results.items(), key=lambda x: x[1]):
        print(f"{model} : {loss}")
        
    
    
    