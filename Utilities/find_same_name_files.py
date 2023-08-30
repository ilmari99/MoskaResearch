import os
import argparse

import os

def find_duplicate_files(folder_one, folder_two, remove=False):
    """
    Find duplicate files in two folders.

    Args:
        folder_one (str): Path to the first folder.
        folder_two (str): Path to the second folder.
        remove (bool, optional): Whether to remove the duplicate files from folder_two. Defaults to False.

    Returns:
        None
    """
    files_in_one = set(os.listdir(folder_one))
    files_in_two = set(os.listdir(folder_two))
    duplicate_files = files_in_one.intersection(files_in_two)
    print(f"Number of duplicate files: {len(duplicate_files)}")
    if not remove:
        return
    for i, file in enumerate(duplicate_files):
        os.remove(folder_two + file)
        if i % 10 == 0:
            print(f"{i/len(duplicate_files)*100}% complete")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find duplicate files in two folders")
    parser.add_argument("folder_one",type=str,help="First folder")
    parser.add_argument("folder_two",type=str,help="Second folder")
    parser.add_argument("--remove",action="store_true",help="Remove duplicate files from second folder")
    args = parser.parse_args()
    find_duplicate_files(args.folder_one,args.folder_two,args.remove)