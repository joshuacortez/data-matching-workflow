import os
import json

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

def make_saved_files_directories(parent_directory = ".."):
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
    prototype_notebooks_path = os.path.join(parent_directory,"prototype_notebooks")
    assert os.path.exists(prototype_notebooks_path), "Need to copy prototype folders first!"
    pass

def check_notebook_files(parent_directory = ".."):
    # parent_directory contains all the data matching files
    config_filepath = os.path.join(parent_directory, "saved_files/config.json")
    assert os.path.exists(config_filepath), "Need to create config file first!"
    with open(config_filepath, "r") as f:
        config = json.load(f)
    
    dataset_names = list(config["datasets"].keys())

    dedup_names = config["dedup-datasets"]
    dedup_names = ["dedup-{}".format(name) for name in dedup_names]

    rl_names = config["rl-datasets"]
    rl_names = ["rl-{}_{}".format(name_1, name_2) for name_1, name_2 in rl_names]
    
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

def get_filepath_mapping(task_name, dataset_name, saved_files_path):
    filepath_mapping = {
        "raw_data":"{}/raw_data.csv".format(dataset_name),
        "unlabeled_data":"{}/unlabeled_data.json".format(dataset_name),
        "unlabeled_data_no_exact":"{}/unlabeled_data_no_exact.json".format(dataset_name),
        "cleaned_strings_folder":"{}/cleaned_strings_checks/".format(dataset_name),
        "value_counts_folder":"{}/value_counts_checks/".format(dataset_name),

        "model_settings":"{}/training_output/model_settings".format(task_name),
        "labeled_data":"{}/training_output/labeled_data.json".format(task_name),
        "blocks":"{}/blocking_output/blocks.csv".format(task_name),
        "model_weights":"{}/output/model_weights.csv".format(task_name),
        "mapped_records":"{}/output/mapped_records.csv".format(task_name),
        "labeled_pair_ids":"{}/output/labeled_pair_ids.csv".format(task_name),
        "cluster_canonical":"{}/output/cluster_canonical.csv".format(task_name),
        "model_settings_test_split":"{}/test_split_output/model_settings".format(task_name),
        "labeled_data_test_split":"{}/test_split_output/labeled_data.json".format(task_name),

    }

    filepath_mapping = {key:os.path.join(saved_files_path, val) for key,val in filepath_mapping.items()}

    return filepath_mapping

def get_filepath(task_name, file_type, saved_files_path):
    assert os.path.exists(saved_files_path), "saved_files_path doesn't exist!"
    
    task_type, dataset_name = task_name.split("-")

    file_type_dataset_list = ["raw_data", "unlabeled_data", "unlabeled_data_no_exact", "cleaned_strings_folder", "value_counts_folder"]

    if (task_type == "rl") & (file_type in file_type_dataset_list):
        dataset_1_name, dataset_2_name = dataset_name.split("_")
        filepath_mapping_1 = get_filepath_mapping(task_name, dataset_1_name, saved_files_path) 
        filepath_mapping_2 = get_filepath_mapping(task_name, dataset_2_name, saved_files_path) 

        filepath_1 = filepath_mapping_1[file_type]
        filepath_2 = filepath_mapping_2[file_type]
        
        return filepath_1, filepath_2

    else:
        filepath_mapping = get_filepath_mapping(task_name, dataset_name, saved_files_path)
        filepath = filepath_mapping[file_type]

        return filepath        

def get_dataset_info(task_name, info_type, saved_files_path):
    task_type = task_name.split("-")[0]
    assert task_type in ["preprocess", "dedup", "rl"], "filename should start with either preprocess, dedup, or rl"
    with open(os.path.join(saved_files_path,"config.json"), "r") as json_file:
        config = json.load(json_file)
    
    dataset_name = task_name.split("-")[1]

    if (task_type == "dedup") or (task_type == "preprocess"):
        info = config["datasets"][dataset_name][info_type]
        return info
    if task_type == "rl":
        dataset_1_name, dataset_2_name = dataset_name.split("_")

        info_1 = config["datasets"][dataset_1_name][info_type]
        info_2 = config["datasets"][dataset_2_name][info_type]

        return info_1, info_2

def get_task_info(task_name, info_type, saved_files_path):
    # the output dictionary is what is passed to the deduper or linker

    with open(os.path.join(saved_files_path,"config.json"), "r") as json_file:
        config = json.load(json_file)

    task_fields = config["tasks"][task_name][info_type]
    return task_fields

def check_is_data_source_deduped(task_name, saved_files_path):
    task_type, dataset_name = task_name.split("-")
    assert task_type == "rl", "relevant only for record linkage"
    with open(os.path.join(saved_files_path,"config.json"), "r") as json_file:
        config = json.load(json_file)

    task_names = list(config["tasks"].keys())
    dedup_names = [name.split("-")[1] for name in task_names if "dedup-" in name]
    dataset_1_name, dataset_2_name = dataset_name.split("_")
    dataset_1_has_dedup = (dataset_1_name in dedup_names)
    dataset_2_has_dedup = (dataset_2_name in dedup_names)

    if dataset_1_has_dedup and dataset_2_has_dedup:
        is_data_source_deduped = True
    elif dataset_1_has_dedup != dataset_2_has_dedup:
        raise Exception("one dataset has dedup while another one doesn't. both should have or neither")
    else:
        is_data_source_deduped = False

    return is_data_source_deduped

def get_proper_unlabeled_data_filepath(task_name, saved_files_path):
    pre_cluster = get_dataset_info(task_name, "pre_cluster_exact_matches", saved_files_path)

    if isinstance(pre_cluster, tuple):
        pre_cluster_bool_1, pre_cluster_bool_2 = pre_cluster

        if pre_cluster_bool_1:
            filepath_1, _ = get_filepath(task_name, "unlabeled_data_no_exact", saved_files_path)
        else:
            filepath_1, _ = get_filepath(task_name, "unlabeled_data", saved_files_path)

        if pre_cluster_bool_2:
            _, filepath_2 = get_filepath(task_name, "unlabeled_data_no_exact", saved_files_path)
        else:
            _, filepath_2 = get_filepath(task_name, "unlabeled_data", saved_files_path)

        return filepath_1, filepath_2

    else:
        if pre_cluster:
            filepath = get_filepath(task_name, "unlabeled_data_no_exact", saved_files_path)
        else:
            filepath = get_filepath(task_name, "unlabeled_data", saved_files_path)

        return filepath

