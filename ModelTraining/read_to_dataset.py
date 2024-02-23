import os
import random
import tensorflow as tf
from tqdm import tqdm
import multiprocessing as mp

""" Create a TensorFlow dataset from (multiple) folders of vector files.
"""

MISC_PARTS = list(range(0,5)) + list(range(161,170))
CARD_PARTS = list((n for n in range(0,430) if n not in MISC_PARTS))

def read_to_dataset(paths, add_channel=False, shuffle_files=True) -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    file_paths = [os.path.join(path, file) for path in paths for file in os.listdir(path) if os.path.isdir(path)]
    if shuffle_files:
        random.shuffle(file_paths)

    print("Found {} files".format(len(file_paths)))
    def txt_line_to_tensor(x):
        s = tf.strings.split(x, sep=",")
        s = tf.strings.to_number(s, out_type=tf.float32)
        return (s[:-1], s[-1])

    def ds_maker(x):
        ds = tf.data.TextLineDataset(x, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        ds = ds.map(txt_line_to_tensor,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
        return ds
    
    
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.interleave(ds_maker,
                                cycle_length=tf.data.experimental.AUTOTUNE,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                deterministic=False)

    # Add a channel dimension if necessary
    if add_channel:
        dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset, len(file_paths)

def combine_folder(files, dest_folder = "combined", ind=0, restore_files=True):
    # Move the files to the temp folder
    temp_folder = f"temp_folder{ind}"
    os.makedirs(temp_folder, exist_ok=True)
    file_to_dest = {}
    for file in files:
        file_to_dest[file] = os.path.join(temp_folder, os.path.basename(file))
        os.system(f"mv {file} {file_to_dest[file]}")
    # Create the destination folder
    os.makedirs(dest_folder, exist_ok=True)
    
    # Combine the files
    print(f"Running: ./Utilities/combine_vector_files_to_one {temp_folder}/ {dest_folder}/data{ind}.csv {len(files)}")
    os.system(f"./Utilities/combine_vector_files_to_one {temp_folder}/ {dest_folder}/data{ind}.csv {len(files)}")
    
    if restore_files:
        # Move the files back to the original folder
        for file in files:
            os.system(f"mv {file_to_dest[file]} {file}")
    # Remove the temp folder
    #os.remove(temp_folder)
    os.system(f"rm -r {temp_folder}")
    print(f"Done combining files: index {ind}")
    return

def combine_folder_mp_wrap(args):
    return combine_folder(*args)

def combine_folders_to_n_files(folders, n_files = 20, name="CombinedFiles"):
    """ Combine all files, that are in the folders
    and merge them to create n_files files.
    """
    all_files = []
    for folder in folders:
        all_files.extend([os.path.join(folder, file) for file in os.listdir(folder) if os.path.isdir(folder)])
    random.shuffle(all_files)
    # For a C script we can give a folder, and it will combine all files in the folder to 1 file.
    # To use it in parallel, we move the files, to n_files different folders, and then combine them in parallel
    # We then move the combined files to a new folder, and move the files back to the original folder
        
    
    # Create arguments: The arguments is a list of files, and the index of the combined file
    args = [(all_files[i::n_files], name, i) for i in range(n_files)]
    with mp.Pool(20) as pool:
        pool.map(combine_folder_mp_wrap, args, chunksize=1)
    print("Done combining files")
    
    

        
if __name__ == "__main__":
    combine_folders_to_n_files(["/home/ilmari/python/MoskaResearch/Datasets/FullyRandomDataset300k/Vectors/"], 20,name="FullyRandomDataset300kSharded")
    exit()
    # Load the saved ds
    #if os.path.exists("./data.tfrecord") and True:
    #    ds = tf.data.Dataset.load("./data.tfrecord")

    #    print("Counting elements")
    #    print(ds.reduce(0, lambda x,_: x+1).numpy())
    #    exit()
    
    ds, total = read_to_dataset(**{"paths": ["/home/ilmari/python/MoskaResearch/Datasets/FullyRandomDataset300k/Vectors/"],
                                 "add_channel": False,
                                 "shuffle_files": True}
                                )
    
    step_counter = tf.Variable(0, trainable=False)
    checkpoint_prefix = "./data_checkpoint"
    path = "./data.tfrecord"
    checkpoint_args = {
        "checkpoint_interval": 300000,
        "step_counter": step_counter,
        "directory": checkpoint_prefix,
        "max_to_keep": 5,
    }
    
    tf.data.Dataset.save(ds, path, compression=None, shard_func=None, checkpoint_args=checkpoint_args)





