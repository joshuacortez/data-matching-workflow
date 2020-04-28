# data-matching-workflow
Make it easier to match records with high accuracy. :tada: This is a template workflow for deduplication and record linkage that uses the [dedupe library](https://github.com/dedupeio/dedupe). It uses [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) to train models for matching records.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/joshuacortez/data-matching-workflow/master)

## How this can be useful:question:
Many organizations rely on having an accurate database of records, such as customer records. However, a common issue is that multiple records may actually refer to the same entity, or the same person. In banking for example, a person may sign up for a deposit account, and then sign up for a credit card. However, the bank may not know that the two accounts refer to the same person. Data matching techniques aim to solve that problem. 

*Duplication example with 4 identifier fields (mispelling in Last Name :scream:)*
| Data Source     | First Name | Last Name | Birthday  | Postcode |
|:------------:|:----------:|:---------:|:---------:|:--------:|
| Deposits     | Michael    | Fernandez | 29-Sep-83 | 1512     |
| Credit Cards | Michael    | Fernandes | 29-Sep-83 | 1522     |

## Workflow Overview
This workflow can be split into 4 main tasks. Under the `notebooks` folder, there are 4 subfolders corresponding to each task. Each subfolder contains a set of notebooks for the subtasks. Here's an overview of the main tasks, and their respective subtasks:
- **Preprocessing** (`preprocess`) 
A comprehensive template for data cleaning before the matching proper. Must be used before deduplication and/or record linkage.
  1. *Main Preprocessing* 
  2. *Exact Duplicate Handling (optional)*
       - Use this only if there is rampant exact matches in all fields in your dataset.
- **Deduplication** (`dedup`) Used for identifying duplicates within a set of records.
  1. *Model Training*
      - Train the model using active learning. After training, model will be able to distinguish if two records refer to the same entity/person or not. 
      - Outputs of this subtask are: labeled training data, [blocking predicates](http://www.cs.utexas.edu/~ml/papers/blocking-icdm-06.pdf), and Logistic Regression model weights.
  2. *Classification* 
      - Use the model classify matches for the rest of the dataset.
      - Check the blocking predicates and model weights to interpret the model.
      - Inspect diagnostic measures and plots to check for matching quality.
  3. *Test Set Evaluation*
      - Split the labeled training data into a training and test set.
      - Re-train the model and evaluate on the test set to have a out-of-sample diagnostic measures.
      - Diagnostic measures here complement the diagnostic measures in *Classification*. 
- **Record Linkage** (`rl`) Used for identifying matches between two sets of records. Each set of records must not have duplicates to begin with.
  - These notebooks are similar to those under Deduplication.
  1. *Model Training*
  2. *Classification*
  3. *Test Set Evaluation*
- **Fusion** (`fusion`) Used as a final step after performing record linkage on multiple pairs of records. Creates a unique identifier across all sets of records.

## Example Use Cases
How you would use the workflow depends on the kind of record datasets you have. The steps may vary and here are some examples:
- There is 1 set of records with duplicates.
  1. Preprocess
  2. Perform deduplication
- There are 2 sets of records, A and B. There aren't duplicates within each set, but there are duplicates across sets.
  1. Preprocess each of A and B
  2. Perform recordlinkage for A and B
- There are 3 sets of records: A, B, and C. There are be duplicates within each set, and there are duplicates across sets. 
  1. Preprocess each of A, B and C. 
  2. Perform duplication for A, B, and C. 
  3. Perform record linkage for 3 pairs: (A,B), (B,C), and (C,A). 
  4. Perform fusion on the 3 linked pairs. 

## Sample Datasets
Sample datasets were used from the [recordlinkage library](https://github.com/J535D165/recordlinkage) to demostrate workflow examples. For deduplication, the `febrl3` dataset was used. For record linkage, the `febrl4` dataset was used (and split into two sets: `febrl4a` and `febrl4b`).

These datasets are available under the `saved_files` folder:
  - `saved_files/febrl3`
  - `saved_files/febrl4a`
  - `saved_files/febrl4b`

## How to Set Up for Your Own Datasets :wrench:
1. Edit `config.json`
- Edit the config file to align with the field names and data types of your datasets. A sample template is provided using the sample datasets.
- Under `datasets`
  - The key should be the dataset name. The value contains the following:
  - `numeric_fields` contains the field names that should be numeric (e.g. `float` or `int`)
  - `date_fields` contains the field names that are dates
  - `primary_key` the field name of the presumed unique identifier
  - `pre_cluster_exact_matches` (`bool` type) is `true` if records with exact matches in all fields should be pre-clustered before deduplication. 
    - Set this to `true` if there is rampant exact matches in all fields in your dataset. Otherwise, leave it at the default `false`. 
- Under `tasks`
  - The key should be in the form `<task_name>-<dataset_name>` where the `task_name` is either `dedup` or `rl`. The value contains the following:
  - `recall_train` (`float` type) is a parameter used for learning the blocking predicates. This parameter ranges from 0.0 to 1.0. See [dedupe's API](https://github.com/dedupeio/dedupe/blob/master/dedupe/api.py) for more detail. 
    - This parameter should only be decreased (e.g. set to 0.9) if there are too many blocks later on after model training. Too many blocks generated by the trained model can lead to an out of memory error during clustering.
  - `fields` contains the data types for each field. See [dedupe's variable definition documentation](https://docs.dedupe.io/en/latest/Variable-definition.html) for more detail.
2. Setting up the folder structure
  - The notebooks are sensitive to the filename conventions and the folder structure because that's how they know which filepaths to read and write from. It's important to set this up.
  - Make sure you've already filled out the `datasets` section of `config.json` (filling out `tasks` section is not necessary to set up the folder structure).
  - Make sure you have the `notebooks` folder with the template notebook files.
    - The `notebooks` folder should be in the same directory as `saved_files` and the `python_scripts` folders. 
  - Run this script: `python_scripts/init_files_folders.py`. This automatically creates the necessary folders and copies the template Jupyter notebooks.
    - You may also opt to do this yourself although it may be more cumbersome.
  - Copy the raw dataset/s in the appropriate folder/s in `saved_files ` Place it under the `<dataset_name>` folder with filename `raw_data.csv`.
  - You're ready to go! Check the `notebooks` folder for the notebook templates pre-prepared for your own datasets.


## Python Dependencies
Refer to `environment.yml`
