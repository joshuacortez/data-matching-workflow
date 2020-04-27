# data-matching-workflow
A template workflow for deduplication and record linkage using the `dedupe` library. Aided with active learning, match records more easily. 

## Workflow Overview
Under the notebooks folder, there are 4 subfolders corresponding to the main data matching tasks.
1. **Preprocessing** (`preprocess`)
  A comprehensive template for data cleaning before the matching proper. Must be used before deduplication and/or record linkage.
2. **Deduplication** (`dedup`)
  Used for identifying duplicates within a set of records.
3. **Record Linkage** (`rl`)
  Used for identifying matches between two sets of records. Each set of records must not have duplicates to begin with.
4. **Fusion** (`fusion`)
  Used as a final step after performing record linkage on multiple pairs of records. Creates a unique identifier across all sets of records.

- Example use case: There are 3 sets of records A, B, and C. There are be duplicates within each set, and there are duplicates across sets. First preprocess each of A, B and C. Then perform duplication for A, B, and C. Afterwards, perform record linkage for 3 pairs: A & B, B & C, and C & A. Lastly, perform fusion on the 3 linked pairs. 

## Sample Datasets
Sample datasets were used from the `recordlinkage` library to demostrate workflow examples. For deduplciation, the `febrl3` dataset was used. For record linkage, the `febrl4` dataset was used (and split into two sets: `febrl4a` and `febrl4b`).

## How to Set Up for Your Own Datasets
1. Editing `config.json`
Edit the config file to align with the field names and data types of your dataset.
  - Under `datasets`
    - The key should be the dataset name. The value contains the following:
    - `numeric_fields` contains the field names that should be numeric (e.g. `float` or `int`)
    - `date_fields` contains the field names that are dates
    - `primary_key` the field name of the presumed unique identifier
    - `pre_cluster_exact_matches` (`bool` type) is `true` if records with exact matches in all fields should be pre-clustered before deduplication. Set to `true` if there is rampant exact matches in all fields in your dataset. This is `false` by default.
 - Under `tasks`
  - The key should be in the form `<task_name>-<dataset_name>` where the `task_name` is either `dedup` or `rl`. The value contains the following:
  - `recall_train` (`float` type) is a parameter used for learning the blocking predicates. This parameter ranges from 0.0 to 1.0. See [`dedupe`'s API](https://github.com/dedupeio/dedupe/blob/master/dedupe/api.py) for more detail. This parameter should only be decreased (i.e. 0.9) if there are too many blocks later on after model training. Too many blocks generated by the trained model can lead to an out of memory error during clustering.
  - `fields` contains the data types for each field. See [`dedupe`'s variable definition documentation](https://docs.dedupe.io/en/latest/Variable-definition.html) for more detail.
2. Setting up the folder structure (TODO)
