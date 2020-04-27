import os
import json
from shutil import copyfile

def make_subfolders(parent_folder, folder_names, subfolder_names):
    # make sure each folder has every subfolder
    assert os.path.exists(parent_folder), "Parent folder {} doesn't exist".format(parent_folder)

    for name in folder_names:
        folder_path = os.path.join(parent_folder, name)
        if not os.path.exists(folder_path):
            print("Creating {}".format(folder_path))
            os.mkdir(folder_path)
        else:
            print("{} already exists".format(folder_path))

        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(folder_path, subfolder_name)
            if not os.path.exists(subfolder_path):
                print("Creating {}".format(subfolder_path))
                os.mkdir(subfolder_path)
            else:
                print("{} already exists".format(subfolder_path))

def check_notebook_filenames(foldername):
    base_foldername = os.path.basename(foldername)
    # check filenames of all jupyter notebooks
    filenames = [i for i in os.listdir(foldername) if ".ipynb_checkpoints" not in i]
    for filename in filenames:
        filename_identifier = "-".join(filename.split("-")[:2])
        assert base_foldername == filename_identifier, "Filename {} doesn't align with foldername {}".format(filename, base_foldername)            

def check_notebook_filenames_all(parent_folder, folder_names, subfolder_names):
    # make sure each folder has every subfolder
    assert os.path.exists(parent_folder), "Parent folder {} doesn't exist".format(parent_folder)

    for name in folder_names:
        folder_path = os.path.join(parent_folder, name)
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(folder_path, subfolder_name)
            check_notebook_filenames(subfolder_path)

def make_directories(parent_directory = ".."):
    # parent_directory contains all the data matching files
    config_filepath = os.path.join(parent_directory, "saved_files/config.json")
    assert os.path.exists(config_filepath), "Need to create config file first!"
    with open(config_filepath, "r") as f:
        config = json.load(f)
    
    dataset_names = list(config["datasets"].keys())

    task_names = list(config["tasks"].keys())
    dedup_names = [name for name in task_names if "dedup-" in name]
    rl_names = [name for name in task_names if "rl-" in name]

    print("Making subfolders for saved_files...")
    saved_files_folderpath = os.path.join(parent_directory, "saved_files")
    make_subfolders(parent_folder = saved_files_folderpath,
                    folder_names = dataset_names, 
                    subfolder_names = ["value_counts_checks", "cleaned_strings_checks"]
                    )

    make_subfolders(parent_folder = saved_files_folderpath,
                    folder_names = dedup_names, 
                    subfolder_names = ["training_output", "blocking_output", "output", "test_split_output"]
                    )
 
    make_subfolders(parent_folder = saved_files_folderpath,
                    folder_names = rl_names,
                    subfolder_names = ["training_output", "blocking_output", "output", "test_split_output"]
                    )

    make_subfolders(parent_folder = saved_files_folderpath,
                    folder_names = ["fusion"],
                    subfolder_names = []
                    )

    print("Making subfolders for notebooks...")
    notebooks_folderpath = os.path.join(parent_directory, "notebooks")
    if not os.path.exists(notebooks_folderpath):
        os.mkdir(notebooks_folderpath)

    make_subfolders(parent_folder = notebooks_folderpath,
                    folder_names = ["preprocess"],
                    subfolder_names = ["preprocess-{}".format(i) for i in dataset_names]
                    )

    make_subfolders(parent_folder = notebooks_folderpath,
                    folder_names = ["dedup"],
                    subfolder_names = dedup_names
                    )

    make_subfolders(parent_folder = notebooks_folderpath,
                    folder_names = ["rl"],
                    subfolder_names = rl_names
                    )

    make_subfolders(parent_folder = notebooks_folderpath,
                    folder_names = ["fusion"],
                    subfolder_names = []
                    )

def copy_prototype_notebooks(parent_directory = ".."):
    # parent_directory contains all the data matching files
    proto_notebooks_path = os.path.join(parent_directory,"prototype_notebooks")
    assert os.path.exists(proto_notebooks_path), "Need to copy prototype folders first!"

    config_filepath = os.path.join(parent_directory, "saved_files/config.json")
    assert os.path.exists(config_filepath), "Need to create config file first!"
    with open(config_filepath, "r") as f:
        config = json.load(f)
    
    dataset_names = list(config["datasets"].keys())

    task_names = list(config["tasks"].keys())
    dedup_names = [name for name in task_names if "dedup-" in name]
    rl_names = [name for name in task_names if "rl-" in name]

    # get all prototype filepaths first
    proto_preprocess_path = os.path.join(proto_notebooks_path, "preprocess/preprocess-febrl3")
    proto_preprocess_ntbks = os.listdir(proto_preprocess_path)
    proto_preprocess_ntbks = [x for x in proto_preprocess_ntbks if (x != ".ipynb_checkpoints") and ("TEST" not in x)]

    proto_dedup_path = os.path.join(proto_notebooks_path, "dedup/dedup-febrl3")
    proto_dedup_ntbks = os.listdir(proto_dedup_path)
    proto_dedup_ntbks = [x for x in proto_dedup_ntbks if (x != ".ipynb_checkpoints") and ("TEST" not in x)]
    
    proto_rl_path = os.path.join(proto_notebooks_path, "rl/rl-febrl4a_febrl4b")
    proto_rl_ntbks = os.listdir(proto_rl_path)
    proto_rl_ntbks = [x for x in proto_rl_ntbks if (x != ".ipynb_checkpoints") and ("TEST" not in x)]
 
    proto_fusion_path = os.path.join(proto_notebooks_path, "fusion")
    proto_fusion_ntbks = os.listdir(proto_fusion_path)
    proto_fusion_ntbks = [x for x in proto_fusion_ntbks if (x != ".ipynb_checkpoints") and ("TEST" not in x)]

    # copy all prototype notebooks to corresponding notebooks

    for dataset_name in dataset_names:
        for source_ntbk in proto_preprocess_ntbks:
            target_ntbk = source_ntbk.replace("febrl3", dataset_name)

            source_path = os.path.join(proto_preprocess_path, source_ntbk)
            target_path = os.path.join(parent_directory, "notebooks/preprocess/preprocess-{}".format(dataset_name), 
                                        target_ntbk)
            if os.path.exists(target_path):
                print("{} already exists".format(target_path))
            else:
                print("Copying prototype preprocessing ntbk {}".format(source_path))
                copyfile(source_path, target_path)
        
    for dedup_name in dedup_names:
        for source_ntbk in proto_dedup_ntbks:
            target_ntbk = source_ntbk.replace("dedup-febrl3", dedup_name)

            source_path = os.path.join(proto_dedup_path, source_ntbk)
            target_path = os.path.join(parent_directory, "notebooks/dedup/{}".format(dedup_name), target_ntbk)
            if os.path.exists(target_path):
                print("{} already exists".format(target_path))
            else:
                print("Copying prototype deduping ntbk {}".format(source_path))
                copyfile(source_path, target_path)

    for rl_name in rl_names:
        for source_ntbk in proto_rl_ntbks:
            target_ntbk = source_ntbk.replace("rl-febrl4a_febrl4b", rl_name)

            source_path = os.path.join(proto_rl_path, source_ntbk)
            target_path = os.path.join(parent_directory, "notebooks/rl/{}".format(rl_name), target_ntbk)
            if os.path.exists(target_path):
                print("{} already exists".format(target_path))
            else:
                print("Copying prototype rl ntbk {}".format(source_path))
                copyfile(source_path, target_path)

    for source_ntbk in proto_fusion_ntbks:
        target_ntbk = source_ntbk
        source_path = os.path.join(proto_fusion_path, source_ntbk)
        target_path = os.path.join(parent_directory, "notebooks/fusion", target_ntbk)

        if os.path.exists(target_path):
            print("{} already exists".format(target_path))
        else:
            print("Copying prototype fusion ntbk {}".format(source_path))
            copyfile(source_path, target_path)

def check_notebook_files(parent_directory = ".."):
    # parent_directory contains all the data matching files
    config_filepath = os.path.join(parent_directory, "saved_files/config.json")
    assert os.path.exists(config_filepath), "Need to create config file first!"
    with open(config_filepath, "r") as f:
        config = json.load(f)
    
    dataset_names = list(config["datasets"].keys())

    task_names = list(config["tasks"].keys())
    dedup_names = [name for name in task_names if "dedup-" in name]
    rl_names = [name for name in task_names if "rl-" in name]
    
    print("Checking filenames in notebook subfolders...")
    notebooks_folderpath = os.path.join(parent_directory, "notebooks")
    assert os.path.exists(notebooks_folderpath), "no notebooks folder found!"

    check_notebook_filenames_all(
                    parent_folder = notebooks_folderpath,
                    folder_names = ["preprocess"],
                    subfolder_names = ["preprocess-{}".format(i) for i in dataset_names]
                    )

    check_notebook_filenames_all(
                    parent_folder = notebooks_folderpath,
                    folder_names = ["dedup"],
                    subfolder_names = dedup_names
                    )

    check_notebook_filenames_all(
                    parent_folder = notebooks_folderpath,
                    folder_names = ["rl"],
                    subfolder_names = rl_names
                    )

    print("Notebook filenames all correct!")

if __name__ == "__main__":
    notice = """
            Run this script only after doing the following in the main project folder:
            1. Have a python_scripts folder with the .py files (this file itself should be inside).
            2. Have a saved_files folder with a filled out config.json file.
            3. Have a notebooks folder.
            4. Have a prototype_notebooks folder filled with prototype notebooks that use the febrl dataset.
            Have you done the steps above already? (y/n)
            """ 
    user_input = ""
    valid_response = False
    while not valid_response:
        print(notice)
        user_input = input("Answer:")
        if user_input in ["y", "n"]:
            valid_response = True
        
        if user_input == "y":
            make_directories()
            copy_prototype_notebooks()
            check_notebook_files()
        elif user_input == "n":
            print("Run this script again after doing the above mentioned steps.")
        else:
            print("Incorrect input. Only type y or n.")