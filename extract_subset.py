import os
import shutil


def extract_subset(in_dir_path, out_dir_path):
    # Read the text file with the paths to be moved
    txt_path = in_dir_path + '/Easy.txt'
    with open(txt_path, 'r') as file:
        to_keep_paths = file.read().split('\n')[:-1]

    # Filter out some unwanted paths
    to_keep_paths = [path for path in to_keep_paths if not 'LR_' in path]

    # Search for and move images to the new (non-existing) folder
    os.mkdir(out_dir_path)
    for root_path, dir_paths, _ in os.walk(in_dir_path):
        for dir_path in dir_paths:
            if dir_path in to_keep_paths:
                to_path = os.path.join(out_dir_path, dir_path)
                os.mkdir(to_path)
                for root_sub_path, _, files in os.walk(os.path.join(root_path, dir_path, 'light_mask')):
                    for file in files:
                        from_path = os.path.join(root_sub_path, file)
                        shutil.move(from_path, to_path)

    # Delete the original folder
    shutil.rmtree(in_dir_path)


if __name__ == '__main__':
    # Example:
    # python /.../extract_subset.py /.../rear_signal_dataset /.../rear_signal_dataset_filtered

    # Call the function with the parameters
    in_dir_fpth = 'rear_signal_dataset/rear_signal_dataset'
    out_dir_fpth = 'rear_signal_dataset_filtered'
    extract_subset(in_dir_fpth, out_dir_fpth)
