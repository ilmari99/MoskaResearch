import os
import random
import argparse
import multiprocessing as mp


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

def combine_folders_to_n_files(folders, n_files = 20, name="CombinedFiles", keep_original_files=True):
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
    args = [(all_files[i::n_files], name, i, keep_original_files) for i in range(n_files)]
    with mp.Pool(20) as pool:
        pool.map(combine_folder_mp_wrap, args, chunksize=1)
    print("Done combining files")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine files')
    
    parser.add_argument("--folders", nargs="*", type=str, help="The folders to combine", required=True)
    parser.add_argument("--n_files", type=int, help="The number of files to combine to", default=20)
    parser.add_argument("--output_folder", type=str, help="The name of the combined files", default="CombinedFiles")
    parser.add_argument("--keep_original_files", action="store_true", help="Keep the original files", default=True)
    
    folders = parser.datasets
    #"Vectors" folder
    folders = [path + os.sep + "Vectors" for path in folders if not path.endswith("Vectors")]
    
    combine_folders_to_n_files(folders, n_files = parser.n_files, name=parser.output_folder, keep_original_files=parser.keep_original_files)
