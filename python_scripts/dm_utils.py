import networkx as nx
import csv
import pandas as pd
import itertools
import json
import dedupe
from itertools import combinations,product
import sys
import os
import numpy as np
from affinegap import normalizedAffineGapDistance
import simplejson
from tqdm import tqdm
import tempfile
from dedupe.clustering import cluster as dedupe_cluster

import dm_file_checker

def get_deduper_probs_and_threshold(deduper, unlabeled_data, blocked_data = None, recall_weight = 1):
    if blocked_data is None:
        pairs = deduper.pairs(unlabeled_data)
    else:
        pairs = itertools.chain.from_iterable(get_blocked_pairs(deduper, blocked_data))

    probs = dedupe.core.scoreDuplicates(pairs,
                                       deduper.data_model,
                                       deduper.classifier,
                                       deduper.num_cores)['score']

    # the memory mapped file location of the scored records
    temp_filename = probs.filename

    probs = probs.copy()
    probs.sort()
    probs = probs[::-1]

    # delete the memory mapped file so it won't clog the disk
    os.remove(temp_filename)

    expected_dupes = np.cumsum(probs)

    recall = expected_dupes / expected_dupes[-1]
    precision = expected_dupes / np.arange(1, len(expected_dupes) + 1)

    score = recall * precision / (recall + recall_weight ** 2 * precision)

    i = np.argmax(score)

    print('Maximum expected recall and precision')
    print('recall: {:.2f}%'.format(recall[i]*100))
    print('precision: {:.2f}%'.format(precision[i]*100))
    print('With threshold: {:.2f}%'.format(probs[i]*100))

    return probs, probs[i]

def get_linker_probs_and_threshold(linker, unlabeled_data_1, unlabeled_data_2, blocked_data = None, recall_weight = 1):
    if blocked_data is None:
        pairs = linker.pairs(unlabeled_data_1, unlabeled_data_2)
    else:
        pairs = itertools.chain.from_iterable(get_blocked_pairs(linker, blocked_data))

    probs = dedupe.core.scoreDuplicates(pairs,
                                       linker.data_model,
                                       linker.classifier,
                                       linker.num_cores)['score']

    # the memory mapped file location of the scored records
    temp_filename = probs.filename

    probs = probs.copy()
    probs.sort()
    probs = probs[::-1]

    # delete the memory mapped file so it won't clog the disk
    os.remove(temp_filename)

    expected_dupes = np.cumsum(probs)

    recall = expected_dupes / expected_dupes[-1]
    precision = expected_dupes / np.arange(1, len(expected_dupes) + 1)

    score = recall * precision / (recall + recall_weight ** 2 * precision)

    i = np.argmax(score)

    print('Maximum expected recall and precision')
    print('recall: {:.2f}%'.format(recall[i]*100))
    print('precision: {:.2f}%'.format(precision[i]*100))
    print('With threshold: {:.2f}%'.format(probs[i]*100))

    return probs, probs[i]

def get_model_weights(deduper_or_linker):

    fields = [field.name for field in deduper_or_linker.data_model._variables]
    model_weights = sorted(list(zip(fields, deduper_or_linker.classifier.weights)), key = lambda x: x[1], reverse = False)

    model_weights = pd.DataFrame(model_weights, columns = ["variable", "logistic_reg_weight"])

    return model_weights

def map_cluster_ids(deduper, unlabeled_data, threshold, hard_threshold = 0.0,
                    blocked_data = None, canonicalize = True, numeric_fields = None,
                    cluster_id_tag = None, 
                    mapped_records_filepath = None,
                    cluster_canonical_filepath = None):

    # BADLY NEED TO REFACTOR THIS
    """
        Function that maps record ids to cluster ids
        Parameters
        ----------
        deduper : dedupe.Deduper
            A trained instance of dedupe.
        unlabeled_data : dict
            The dedupe formatted data dictionary.
        threshold : dedupe.Threshold
            The threshold used for clustering.
        hard_threshold: float
            Threshold for record pair scores that will be included in the clustering
        canonicalize : bool or list, default False
            Option that provides the canonical records as additional columns.
            Specifying a list of column names only canonicalizes those columns.
        numeric_fields: list of str, default None
            Specify which fields are numeric
        cluster_id_tag: str, default None
            Additional tag for distinguishing the cluster id of different datasets
        Returns
        -------
        mapped_records
            A dataframe storing the mapping from cluster_id to record_id
        cluster_canonicals
            A dataframe storing the canonical representation per cluster_id
    """

    assert (hard_threshold < 1) and (hard_threshold >= 0), "hard_threshold should less than 1 at at least 0.0"
    
    if mapped_records_filepath is not None:
        with open(mapped_records_filepath, "w", newline = "") as f:
            mapped_records_header = ["record id", "cluster id", "confidence score", "cluster type"]
            writer = csv.DictWriter(f, fieldnames = mapped_records_header, quoting = csv.QUOTE_ALL)
            writer.writeheader()

    if canonicalize:
        if cluster_canonical_filepath is not None:
            with open(cluster_canonical_filepath, "w", newline = "") as f:
                cluster_canonical_header = [field.field for field in deduper.data_model.primary_fields]
                cluster_canonical_header.append("cluster id")
                writer = csv.DictWriter(f, fieldnames = cluster_canonical_header, quoting = csv.QUOTE_ALL)
                writer.writeheader()
    else:
        assert cluster_canonical_filepath is None, "can't have canonicalize be False if cluster_canonical_filepath exists"

    # ## Clustering
    if blocked_data is None:
        pairs = deduper.pairs(unlabeled_data)
    else:
        pairs = itertools.chain.from_iterable(get_blocked_pairs(deduper, blocked_data))

    pair_scores = deduper.score(pairs)
    pair_scores = pair_scores[pair_scores["score"] > hard_threshold]
    clustered_dupes = deduper.cluster(pair_scores, threshold)

    if numeric_fields is not None:
        assert isinstance(numeric_fields, list)
    
    mapped_records = []
    cluster_canonicals = []
    record_ids_in_clusters = []

    # assign cluster ids to record ids
    i = 0
    print("Mapping cluster ids...")
    for cluster in tqdm(clustered_dupes):
        i += 1
        cluster_id = "cl-{}".format(i)
        if cluster_id_tag is not None:
            cluster_id = "{}-{}".format(cluster_id_tag, cluster_id)
        id_set, scores = cluster

        if canonicalize:
            cluster_data = [unlabeled_data[i] for i in id_set]
            canonical_rep = get_canonical_rep(cluster_data, numeric_fields = numeric_fields)
            canonical_rep["cluster id"] = cluster_id

            if cluster_canonical_filepath is not None:
                with open(cluster_canonical_filepath, "a") as f:
                    writer = csv.DictWriter(f, fieldnames = cluster_canonical_header, quoting = csv.QUOTE_ALL)
                    writer.writerow(canonical_rep)
            else:
                cluster_canonicals.append(canonical_rep)

        for record_id, score in zip(id_set, scores):
            record_dict = {
                "record id": record_id,
                "cluster id": cluster_id,
                "confidence score": score,
                "cluster type":'dup'
            }
            
            if mapped_records_filepath is not None:
                with open(mapped_records_filepath, "a", newline = "") as f:
                    writer = csv.DictWriter(f, fieldnames = mapped_records_header, quoting = csv.QUOTE_ALL)
                    writer.writerow(record_dict)
            else:
                mapped_records.append(record_dict)
            
            record_ids_in_clusters.append(record_id)

    record_ids_in_clusters = set(record_ids_in_clusters)
    solo_ids = list(set(unlabeled_data.keys()).difference(record_ids_in_clusters))

    # assign solo ids to record ids
    print("Mapping solo record ids...")
    for record_id in tqdm(solo_ids):
        i += 1
        cluster_id = "cl-{}".format(i)
        if cluster_id_tag is not None:
            cluster_id = "{}-{}".format(cluster_id_tag, cluster_id)
        record_dict = {
            "record id":record_id,
            "cluster id":cluster_id,
            "confidence score":None,
            "cluster type":'solo'
        }
        
        mapped_records.append(record_dict)

    if mapped_records_filepath is None:
        mapped_records = pd.DataFrame(mapped_records)
    else:
        with open(mapped_records_filepath, "a", newline = "") as f:
            writer = csv.DictWriter(f, fieldnames = mapped_records_header, quoting = csv.QUOTE_ALL)
            writer.writerows(mapped_records)
        mapped_records = None

    if cluster_canonical_filepath is None:
        cluster_canonicals = pd.DataFrame(cluster_canonicals)
    else:
        cluster_canonicals = None

    # delete temporary file generated for pair_scores	
    try:	
        mmap_file = pair_scores.filename	
        del pair_scores	
        os.remove(mmap_file)	
    except AttributeError:	
        pass

    if canonicalize:
        return mapped_records, cluster_canonicals
    else:
        return mapped_records

def abs_distance(x,y):
    return np.abs(x-y)

def get_canonical_rep(record_cluster, numeric_fields = None):
    """
    Given a list of records within a duplicate cluster, constructs a
    canonical representation of the cluster by finding canonical
    values for each field
    """
    canonical_rep = {}

    keys = record_cluster[0].keys()

    if numeric_fields is None:
        numeric_fields = []

    for key in keys:
        key_values = []

        # difference distance functions for numeric and non-numeric fields
        if key in numeric_fields:
            comparator = abs_distance
        else:
            comparator = normalizedAffineGapDistance
        for record in record_cluster:
            # assume non-empty values always better than empty value
            # for canonical record
            if record[key]:
                key_values.append(record[key])
        if key_values:
            canonical_rep[key] = dedupe.canonical.getCentroid(key_values, comparator)
        else:
            canonical_rep[key] = ''

    return canonical_rep

def get_linked_ids(linker, unlabeled_data_1, unlabeled_data_2, threshold, hard_threshold = 0.0, blocked_data = None, 
                    mapped_records_filepath = None, constraint = "one-to-one"):
    # BADLY NEED TO REFACTOR THIS
    """
      constraint: What type of constraint to put on a join.
                        'one-to-one'
                              Every record in data_1 can match at most
                              one record from data_2 and every record
                              from data_2 can match at most one record
                              from data_1. This is good for when both
                              data_1 and data_2 are from different
                              sources and you are interested in
                              matching across the sources. If,
                              individually, data_1 or data_2 have many
                              duplicates you will not get good
                              matches.
                        'many-to-one'
                              Every record in data_1 can match at most
                              one record from data_2, but more than
                              one record from data_1 can match to the
                              same record in data_2. This is good for
                              when data_2 is a lookup table and data_1
                              is messy, such as geocoding or matching
                              against golden records.
                        'many-to-many'
                              Every record in data_1 can match
                              multiple records in data_2 and vice
                              versa. This is like a SQL inner join.
    """

    if mapped_records_filepath is not None:
        with open(mapped_records_filepath, "w", newline = "") as f:
            mapped_records_header = ["record id 1", "record id 2", "confidence score", "link type"]
            writer = csv.DictWriter(f, fieldnames = mapped_records_header, quoting = csv.QUOTE_ALL)
            writer.writeheader()

    ## link matching
    if blocked_data is None:
        pairs = linker.pairs(unlabeled_data_1, unlabeled_data_2)
    else:
        pairs = itertools.chain.from_iterable(get_blocked_pairs(linker, blocked_data))

    pair_scores = linker.score(pairs)
    pair_scores = pair_scores[pair_scores["score"] > hard_threshold]

    assert constraint in {'one-to-one', 'many-to-one', 'many-to-many'}, (
            '%s is an invalid constraint option. Valid options include '
            'one-to-one, many-to-one, or many-to-many' % constraint)
    if constraint == 'one-to-one':
        links = linker.one_to_one(pair_scores, threshold)
    elif constraint == 'many-to-one':
        links = linker.many_to_one(pair_scores, threshold)
    elif constraint == 'many-to-many':
        links = pair_scores[pair_scores['score'] > threshold]
    links = list(links)

    # delete temporary file generated for pair_scores
    try:
        mmap_file = pair_scores.filename
        del pair_scores
        os.remove(mmap_file)
    except AttributeError:
        pass

    mapped_records = []

    ids_with_links_1 = []
    ids_with_links_2 = []

    print("Mapping linked pairs...")
    for record_pair in tqdm(links):
        record_ids, score = record_pair
        pair_dict = {
            "record id 1":record_ids[0],
            "record id 2":record_ids[1],
            "confidence score":score,
            "link type":"dup",
        }
        if mapped_records_filepath is not None:
            with open(mapped_records_filepath, "a", newline = "") as f:
                mapped_records_header = ["record id 1", "record id 2", "confidence score", "link type"]
                writer = csv.DictWriter(f, fieldnames = mapped_records_header, quoting = csv.QUOTE_ALL)
                writer.writerow(pair_dict)
        else:
            mapped_records.append(pair_dict)
        ids_with_links_1.append(record_ids[0])
        ids_with_links_2.append(record_ids[1])

    ids_with_links_1 = set(ids_with_links_1)
    ids_with_links_2 = set(ids_with_links_2)

    # include the records without found links
    ids_without_links_1 = list(set(unlabeled_data_1.keys()).difference(ids_with_links_1))
    ids_without_links_2 = list(set(unlabeled_data_2.keys()).difference(ids_with_links_2))

    print("Mapping unlinked records in dataset 1...")
    for record_id in tqdm(ids_without_links_1):
        pair_dict = {
            "record id 1":record_id,
            "record id 2":None,
            "confidence score":None,
            "link type":"solo",
        }

        mapped_records.append(pair_dict)
    
    print("Mapping unlinked records in dataset 2...")
    for record_id in tqdm(ids_without_links_2):
        pair_dict = {
            "record id 1":None,
            "record id 2":record_id,
            "confidence score":None,
            "link type":"solo",
        }
        mapped_records.append(pair_dict)

    if mapped_records_filepath is None:
        mapped_records = pd.DataFrame(mapped_records)
    else:
        with open(mapped_records_filepath, "a", newline = "") as f:
            mapped_records_header = ["record id 1", "record id 2", "confidence score", "link type"]
            writer = csv.DictWriter(f, fieldnames = mapped_records_header, quoting = csv.QUOTE_ALL)
            writer.writerows(mapped_records)
        mapped_records = None

    return mapped_records


def get_uncertain_clusters(mapped_records_df, threshold = 0.9):
    cluster_means_df = mapped_records_df\
                        .groupby("cluster id")\
                        .mean()\
                        .sort_values(by = "confidence score", ascending = True)
    
    cluster_means_bool = (cluster_means_df["confidence score"] < threshold)
    print("There are {} clusters with mean confidence score lower than {:.1f}% threshold".format(cluster_means_bool.sum(), threshold*100))
    
    uncertain_clusters_dict = cluster_means_df.loc[cluster_means_bool,:].to_dict()["confidence score"]
    
    return uncertain_clusters_dict

def get_pairs_from_uncertain_clusters(mapped_records_df, labeled_id_pairs, threshold = 0.9):
    assert isinstance(labeled_id_pairs, list)

    uncertain_clusters = get_uncertain_clusters(mapped_records_df, threshold = threshold)
    n_uncertain_clusters = len(uncertain_clusters)

    nth_cluster = 0
    for cluster_id, mean_conf_score in uncertain_clusters.items():
        nth_cluster += 1
        pairs_in_cluster = []

        # get record ids in cluster
        ids_in_cluster = mapped_records_df.loc[mapped_records_df["cluster id"] == cluster_id,"record id"].values.tolist()

        # generating record pairs from cluster
        for id_1, id_2 in combinations(ids_in_cluster, 2):
            id_pair = tuple(sorted((id_1,id_2)))
            # if pair is not already tagged, grab data of records
            if id_pair not in labeled_id_pairs:
                pairs_in_cluster.append(id_pair)

        yield ids_in_cluster, pairs_in_cluster, nth_cluster, n_uncertain_clusters, mean_conf_score

def find_ids_of_labeled_data(labeled_data, unlabeled_data):
    labeled_pair_ids = []
    for label in labeled_data.keys():
        assert label in ["distinct", "match"]
        print("Finding ids for {} pairs".format(label))

        data_pairs_list = labeled_data[label]

        for data_pair in tqdm(data_pairs_list):
            try:
                # for backwards compatibility
                record_1, record_2 = data_pair["__value__"]
            except:
                record_1, record_2 = data_pair
            record_1_id = [key for key,val in unlabeled_data.items() if unlabeled_data[key] == record_1]
            record_2_id = [key for key,val in unlabeled_data.items() if unlabeled_data[key] == record_2]
            
            if len(record_1_id) > 1:
                print("Multiple record ids ({}) found for {}".format(len(record_1_id),record_1))
            record_1_id = record_1_id[0]
                
            if len(record_2_id) > 1:
                print("Multiple record ids ({}) found for {}".format(len(record_2_id),record_2))
            record_2_id = record_2_id[0]
            
            labeled_pair = {"record id 1":record_1_id, "record id 2":record_2_id, "label":label}
            labeled_pair_ids.append(labeled_pair)
    
    labeled_pair_ids = pd.DataFrame(labeled_pair_ids, dtype = "str")
    
    return labeled_pair_ids

def find_ids_of_labeled_data_rl(labeled_data, unlabeled_data_1, unlabeled_data_2):
    labeled_pair_ids = []
    for label in labeled_data.keys():
        assert label in ["distinct", "match"]
        print("Finding ids for {} pairs".format(label))
        data_pairs_list = labeled_data[label]
        for data_pair in tqdm(data_pairs_list):
            record_1, record_2 = data_pair
            record_1_id = [key for key,val in unlabeled_data_1.items() if unlabeled_data_1[key] == record_1]
            record_2_id = [key for key,val in unlabeled_data_2.items() if unlabeled_data_2[key] == record_2]
            
            if len(record_1_id) > 1:
                print("Multiple record ids ({}) found for {}".format(len(record_1_id),record_1))
            record_1_id = record_1_id[0]
                
            if len(record_2_id) > 1:
                print("Multiple record ids ({}) found for {}".format(len(record_2_id),record_2))
            record_2_id = record_2_id[0]
            
            labeled_pair = {"record id 1":record_1_id, "record id 2":record_2_id, "label":label}
            labeled_pair_ids.append(labeled_pair)
    
    labeled_pair_ids = pd.DataFrame(labeled_pair_ids, dtype = "str")
    
    return labeled_pair_ids

def consoleLabel_cluster_old(deduper, mapped_records_df, labeled_id_pairs, unlabeled_data, threshold = 0.9):
    '''
    Command line interface for presenting and labeling uncertain clusters by the user
    Argument :
    A deduper object
    '''

    finished = False
    fields = [field.field for field in deduper.data_model.primary_fields]
    assert len(fields) == len(list(set(fields)))

    labeled_pairs = {"distinct":[], "match":[]}

    uncertain_pair_generator = get_pairs_from_uncertain_clusters(mapped_records_df, 
                                                                labeled_id_pairs, 
                                                                threshold = threshold)
    while not finished:
        try:
            ids_in_cluster, pairs_in_cluster, nth_cluster, n_uncertain_clusters, mean_conf_score = next(uncertain_pair_generator)
            records_in_cluster = {i:unlabeled_data[i] for i in ids_in_cluster}

        except StopIteration:
            print("Already tagged all {} uncertain clusters.".format(n_uncertain_clusters))
            print("Finished labeling")
            break
        
        print("Viewing {} out of {} uncertain clusters".format(nth_cluster, n_uncertain_clusters), file = sys.stderr)
        print("Cluster contains {} records".format(len(ids_in_cluster)))
        print("Mean Cluster Score {:.1f}%\n".format(mean_conf_score*100), file = sys.stderr)

        for record_id, record in records_in_cluster.items():
            print("Record {}".format(record_id), file=sys.stderr)
            for field in fields:
                line = "{} : {}".format(field, record[field])
                print(line, file=sys.stderr)
            print(file=sys.stderr)

        user_input = _prompt_records_same()

        if user_input == "y":
            for id_1, id_2 in pairs_in_cluster:
                record_pair = (unlabeled_data[id_1], unlabeled_data[id_2])
                labeled_pairs["match"].append(record_pair)

        elif user_input == "n":
            print("Reviewing pairs in cluster", file=sys.stderr)

            for id_1, id_2 in pairs_in_cluster:
                record_pair = (unlabeled_data[id_1], unlabeled_data[id_2])
                for record in record_pair:
                    for field in fields:
                        line = "{} : {}".format(field, record[field])
                        print(line, file=sys.stderr)
                    print(file=sys.stderr)
                
                user_input = _prompt_records_same()

                if user_input == "y":
                    labeled_pairs["match"].append(record_pair)
                elif user_input == "n":
                    labeled_pairs["distinct"].append(record_pair)
                elif user_input == "f":
                    print("Finished labeling", file=sys.stderr)
                    finished = True
                    break
        
        elif user_input == "f":
            print("Finished labeling", file=sys.stderr)
            finished = True

    deduper.markPairs(labeled_pairs)

def consoleLabel_cluster(deduper, mapped_records_df, labeled_id_pairs, unlabeled_data, 
                        recall = 1.0, threshold = 0.9):
    '''
    Command line interface for presenting and labeling uncertain clusters by the user
    Argument :
    A deduper object
    '''

    finished = False
    fields = [field.field for field in deduper.data_model.primary_fields]
    assert len(fields) == len(list(set(fields)))

    labeled_pairs = {"distinct":[], "match":[]}

    uncertain_pair_generator = get_pairs_from_uncertain_clusters(mapped_records_df, 
                                                                labeled_id_pairs, 
                                                                threshold = threshold)
    while not finished:
        try:
            ids_in_cluster, pairs_in_cluster, nth_cluster, n_uncertain_clusters, mean_conf_score = next(uncertain_pair_generator)
            records_in_cluster = {i:unlabeled_data[i] for i in ids_in_cluster}

        except StopIteration:
            print("Already tagged all {} uncertain clusters.".format(n_uncertain_clusters))
            print("Finished labeling")
            break
        
        print("Viewing {} out of {} uncertain clusters".format(nth_cluster, n_uncertain_clusters), file = sys.stderr)
        print("Cluster contains {} records".format(len(ids_in_cluster)), file = sys.stderr)
        print("Mean Cluster Score {:.1f}%\n".format(mean_conf_score*100), file = sys.stderr)

        for record_id, record in records_in_cluster.items():
            print("Record {}".format(record_id), file=sys.stderr)
            for field in fields:
                line = "{} : {}".format(field, record[field])
                print(line, file=sys.stderr)
            print(file=sys.stderr)

        user_input = _prompt_records_same()

        if user_input == "y":
            for id_1, id_2 in pairs_in_cluster:
                record_pair = (unlabeled_data[id_1], unlabeled_data[id_2])
                labeled_pairs["match"].append(record_pair)
                labeled_id_pairs.append((id_1, id_2))

        elif user_input == "n":
            print("Reviewing pairs in cluster", file=sys.stderr)

            for id_1, id_2 in pairs_in_cluster:
                record_pair = (unlabeled_data[id_1], unlabeled_data[id_2])
                for record in record_pair:
                    for field in fields:
                        line = "{} : {}".format(field, record[field])
                        print(line, file=sys.stderr)
                    print(file=sys.stderr)
                
                pair_user_input = _prompt_records_same()

                if pair_user_input == "y":
                    labeled_pairs["match"].append(record_pair)
                    labeled_id_pairs.append((id_1,id_2))
                elif pair_user_input == "n":
                    labeled_pairs["distinct"].append(record_pair)
                    labeled_id_pairs.append((id_1,id_2))
                elif pair_user_input == "f":
                    print("Finished labeling", file=sys.stderr)
                    finished = True
                    break
        
        elif user_input == "f":
            print("Finished labeling", file=sys.stderr)
            finished = True

        if (user_input == "y") or (user_input == "n"):
            deduper.markPairs(labeled_pairs)
            deduper.train(recall = recall)
            clustering_threshold = deduper.threshold(unlabeled_data, recall_weight=1)
            mapped_records_df = map_cluster_ids(deduper, unlabeled_data, clustering_threshold, canonicalize=False)

            print("Resampling uncertain clusters based on retrained model", file=sys.stderr)
            labeled_pairs = {"distinct":[], "match":[]}
            uncertain_pair_generator = get_pairs_from_uncertain_clusters(mapped_records_df, labeled_id_pairs, threshold = threshold)
            

def _prompt_records_same():
    print("Do these records refer to the same thing?", file = sys.stderr)
    valid_response = False
    user_input = ""
    valid_responses = {"y", "n", "u", "f"}
    while not valid_response:
        prompt = "(y)es / (n)o / (u)nsure / (f)inished"

        print(prompt, file=sys.stderr)
        user_input = input()
        if user_input in valid_responses:
            valid_response = True

    return user_input

def get_clusters_from_links(links, solo_records):
    assert isinstance(links, pd.Index)
    assert isinstance(solo_records, pd.Index)
    
    clusters = nx.Graph(links.tolist())
    clusters = list(nx.connected_components(clusters))
    clusters.extend(solo_records.tolist())
    
    return clusters

def get_deduper_candidate_pairs(deduper, unlabeled_data):
    # gets candidate pairs after indexing
    candidate_records = deduper.pairs(unlabeled_data)
    candidate_records = [(candidate[0][0], candidate[1][0]) for candidate in candidate_records]
    candidate_records = pd.MultiIndex.from_tuples(candidate_records)
    # some candidate records can be placed in more than 1 block, so let's drop duplicates
    candidate_records = candidate_records.drop_duplicates()

    return candidate_records

def get_linker_candidate_pairs(linker, unlabeled_data_1, unlabeled_data_2):
    # gets candidate pairs after indexing
    candidate_records = linker.pairs(unlabeled_data_1, unlabeled_data_2)
    candidate_records = [(candidate[0][0], candidate[1][0]) for candidate in candidate_records]
    candidate_records = pd.MultiIndex.from_tuples(candidate_records)
    # some candidate records can be placed in more than 1 block, so let's drop duplicates
    candidate_records = candidate_records.drop_duplicates()

    return candidate_records

# converts multindex to format preferred by dedupe method
def convert_rl_to_dedupe_candidate_pair(candidate_pairs, unlabeled_data):
    assert isinstance(candidate_pairs, pd.Index)

    output = []
    for rec_id_1, rec_id_2 in candidate_pairs:
        # dedupe candidate pairs must be in the format (record_id, record)
        candidate_1 = (rec_id_1, unlabeled_data[rec_id_1])
        candidate_2 = (rec_id_2, unlabeled_data[rec_id_2])
        candidate_pair = (candidate_1, candidate_2)
        output.append(candidate_pair)

    return output

# converts multiindex to format preferred by linker method
def convert_rl_to_linker_candidate_pair(candidate_pairs, unlabeled_data_1, unlabeled_data_2):
    assert isinstance(candidate_pairs, pd.Index)

    output = []
    for rec_id_1, rec_id_2 in candidate_pairs:
        if rec_id_1 in unlabeled_data_1.keys():
            rec_data_1 = unlabeled_data_1[rec_id_1]
            rec_data_2 = unlabeled_data_2[rec_id_2]

            assert rec_id_1 not in unlabeled_data_2.keys(), "{} key found in both datasets. Keys must be unique".format(rec_id_1)
            assert rec_id_2 not in unlabeled_data_1.keys(), "{} key found in both datasets. Keys must be unique".format(rec_id_2)
        else:
            rec_data_1 = unlabeled_data_2[rec_id_1]
            rec_data_2 = unlabeled_data_1[rec_id_2]

            assert rec_id_2 not in unlabeled_data_2.keys(), "{} found in both datasets. Keys must be unique".format(rec_id_2)
        # record linker candidate pairs must be in the format (record_id, record)
        candidate_1 = (rec_id_1, rec_data_1)
        candidate_2 = (rec_id_2, rec_data_2)
        candidate_pair = (candidate_1, candidate_2)
        output.append(candidate_pair)

    return output

def read_unlabeled_data_json(unlabeled_data_filepath, empty_str_to_none = True, numeric_fields = None):
    with open(unlabeled_data_filepath, "r") as json_file:
        unlabeled_data = json.load(json_file)

    unlabeled_data = pd.DataFrame.from_dict(unlabeled_data, orient = "index")

    if numeric_fields is not None:
        assert isinstance(numeric_fields, list)
        for col in numeric_fields:
            unlabeled_data[col] = unlabeled_data[col].apply(lambda x: x if x == "" else float(x))

    if empty_str_to_none:
        for col in unlabeled_data.columns.tolist():

            empty_str_bool = (unlabeled_data[col] == "")
            print("converting {} empty string values of column {} to None".format(empty_str_bool.sum(), col))
            unlabeled_data.loc[empty_str_bool,col] = None

        # converting NaNs of numeric columns (NaNs introduced because of the previous line) to None
        if numeric_fields is not None:
            for col in numeric_fields:
                not_nan_bool = unlabeled_data[col].notnull()
                print("converting {} NaN values of column {} to None".format((~not_nan_bool).sum(), col))
                unlabeled_data[col] = unlabeled_data[col].where((not_nan_bool), None)

    unlabeled_data = unlabeled_data.to_dict(orient = "index")
    
    return unlabeled_data

def write_canonical_w_solo_unlabeled_data(canonicals_df, mapped_records_df, unlabeled_data,
                                         canonical_w_solo_unlabeled_filepath):
    # will be used for post cluster review, specifically on matching solos to clusters and merging clusters
    # those two steps are based on the cluster canonicals

    # remember to read in this written file using read_unlabeled_data_json later on
    
    canonical_w_solo_data = canonicals_df.set_index("cluster id")\
                                        .to_dict(orient = "index")

    mapped_records_df = mapped_records_df.set_index("record id")
    solo_records = mapped_records_df.loc[mapped_records_df["cluster type"] == "solo",:]\
                                    .index.tolist()
    
    for record_id in solo_records:
        record = unlabeled_data[record_id]
        cluster_id = mapped_records_df.loc[record_id,"cluster id"]
        canonical_w_solo_data[cluster_id] = record
        
    with open(canonical_w_solo_unlabeled_filepath, 'w') as outfile:
        json.dump(canonical_w_solo_data, outfile)

def prepare_training_deduper(deduper, unlabeled_data, labeled_data_filepath, blocked_proportion = 0.5, sample_size = 15_000):
    # If we have training data saved from a previous run of dedupe,
    # look for it and load it in.
    # __Note:__ if you want to train from scratch, delete the labeled_data_filepath
    if os.path.exists(labeled_data_filepath):
        print('reading labeled examples from ', labeled_data_filepath)
        with open(labeled_data_filepath, 'rb') as labeled_data:
            deduper.prepare_training(data = unlabeled_data, training_file = labeled_data,
                                    blocked_proportion = blocked_proportion,
                                    sample_size = sample_size)
    else:
        deduper.prepare_training(data = unlabeled_data, blocked_proportion = blocked_proportion,
                                sample_size = sample_size)

def save_trained_deduper(deduper, labeled_data_filepath, settings_filepath):
    # When finished, save our training to disk
    with open(labeled_data_filepath, 'w') as tf:
        deduper.write_training(tf)

    # Save our weights and predicates to disk.  If the settings file
    # exists, we will skip all the training and learning next time we run
    # this file.
    with open(settings_filepath, 'wb') as sf:
        deduper.write_settings(sf)

def prepare_training_linker(linker, unlabeled_data_1, unlabeled_data_2, labeled_data_filepath, blocked_proportion = 0.5, sample_size = 15_000):
    # If we have training data saved from a previous run of linker,
    # look for it and load it in.
    # __Note:__ if you want to train from scratch, delete the labeled_data_filepath
    if os.path.exists(labeled_data_filepath):
        print('reading labeled examples from ', labeled_data_filepath)
        with open(labeled_data_filepath, 'rb') as labeled_data:
            linker.prepare_training(data_1 = unlabeled_data_1, data_2 = unlabeled_data_2,
                                    training_file = labeled_data,
                                    blocked_proportion = blocked_proportion,
                                    sample_size = sample_size)
    else:
        linker.prepare_training(data_1 = unlabeled_data_1, data_2 = unlabeled_data_2, 
                                blocked_proportion = blocked_proportion,
                                sample_size = sample_size)

def save_trained_linker(linker, labeled_data_filepath, settings_filepath):
    # When finished, save our training to disk
    with open(labeled_data_filepath, 'w') as tf:
        linker.write_training(tf)

    # Save our weights and predicates to disk.  If the settings file
    # exists, we will skip all the training and learning next time we run
    # this file.
    with open(settings_filepath, 'wb') as sf:
        linker.write_settings(sf)

def get_data_of_labeled_pairs(labeled_pairs_df, unlabeled_data):
    df = pd.DataFrame.from_dict(unlabeled_data, orient = "index")

    df_left = df.loc[labeled_pairs_df["record id 1"],:]
    df_left.columns = ["{}_1".format(col) for col in df_left.columns]
    df_left.index.name = "record id 1"
    df_left = df_left.reset_index()

    df_right = df.loc[labeled_pairs_df["record id 2"],:]
    df_right.columns = ["{}_2".format(col) for col in df_right.columns]
    df_right.index.name = "record id 2"
    df_right = df_right.reset_index()


    output = pd.concat([df_left, df_right], axis = 1, sort = False)
    # sort columns
    output = output.sort_index(axis = 1)
    output = output.set_index(["record id 1", "record id 2"])

    label_df = labeled_pairs_df.set_index(["record id 1", "record id 2"])
    output = pd.merge(left = label_df, right = output, left_index = True, right_index = True, how = "inner")
    
    return output

def get_deduped_data(mapped_records_df, canonicals_df, unlabeled_data, none_to_empty_str = True):
    mapped_records_df = mapped_records_df.set_index("record id")
    solo_record_ids = mapped_records_df.loc[mapped_records_df["cluster type"] == "solo","cluster id"].to_dict()
    deduped_data = {cluster_id:unlabeled_data[record_id] for record_id,cluster_id in solo_record_ids.items()}
    deduped_data = pd.DataFrame.from_dict(deduped_data, orient = "index")
    deduped_data.index.name = "cluster id"
    
    canonicals_df = canonicals_df.set_index("cluster id")
    
    # appending the canonicalized cluster representations to the solo records
    deduped_data = deduped_data.append(canonicals_df)

    if none_to_empty_str:
        deduped_data = deduped_data.where((deduped_data.notnull()), "")
    
    return deduped_data

def write_deduper_blocks(deduper, unlabeled_data, blocks_filepath):
    """
    simplify blocks by not writing the record entries, only the ids
    """
    blocks = deduper.pairs(unlabeled_data)

    with open(blocks_filepath, "w", newline = "") as csv_file:
        writer = csv.writer(csv_file, quoting = csv.QUOTE_ALL)
        header = ["block_id", "record_id"]
        writer.writerow(header)

        block_id = 1
        for block in blocks:
            """
            from dedupe source code:
            Each item in a block must be a sequence of record_id, record and the 
            records also must be dictionaries

            but we're only keeping the record_id, not the record here
            """

            for record in block:
                record_id, _, = record
                block_entry = [block_id, record_id]
                writer.writerow(block_entry)
            block_id += 1 

def read_deduper_blocks(unlabeled_data, blocks_filepath):
    # assumes that the records are sorted by block number
    current_block_id = None
    block_records = []

    """
    from dedupe source code:
    Each item in a block must be a sequence of record_id, record, and the 
    records also must be dictionaries
    """

    with open(blocks_filepath, "r") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            block_id, record_id = row["block_id"], row["record_id"]

            if current_block_id == block_id:
                block_records.append((record_id, unlabeled_data[record_id]))
            else:
                if current_block_id is not None:
                    yield block_records
                current_block_id = block_id
                block_records = [(record_id, unlabeled_data[record_id])]
        
        yield block_records

def write_linker_blocks(linker, unlabeled_data_1, unlabeled_data_2, blocks_filepath):
    """
    simplify blocks by not writing the record entries, only the ids
    """
    blocks = linker.pairs(unlabeled_data_1, unlabeled_data_2)

    block_id = 1
    with open(blocks_filepath, "w", newline = "") as csv_file:
        writer = csv.writer(csv_file, quoting = csv.QUOTE_ALL)
        header = ["record_set_num", "block_id", "record_id"]
        writer.writerow(header)

        for block in blocks:
          
            rec_1, rec_2 = block

            rec_1_id, _  = rec_1
            block_entry = ["1", block_id, rec_1_id]
            writer.writerow(block_entry)
            
            rec_2_id, _  = rec_2
            block_entry = ["2", block_id, rec_2_id]
            writer.writerow(block_entry)

            block_id += 1

def read_linker_blocks(unlabeled_data_1, unlabeled_data_2, blocks_filepath):
    # assumes that the records sorted by block number
    block_records = ()
    block_set_1 = []
    block_set_2 = []
    current_block_id = None

    """
    from dedupe source code:
    Each block must be a made up of two sequences, (base_sequence, target_sequence)
    """
    
    with open(blocks_filepath, "r") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            record_set_num, block_id, record_id = row["record_set_num"], row["block_id"], row["record_id"]

            if current_block_id == block_id:
                if record_set_num == "1":
                    block_set_1.append((record_id, unlabeled_data_1[record_id]))
                elif record_set_num == "2":
                    block_set_2.append((record_id, unlabeled_data_2[record_id]))
                else:
                    raise ValueError("record_set_num should only be 1 or 2, but got {}".format(record_set_num))
            else:
                if current_block_id is not None:
                    block_records = (block_set_1, block_set_2)
                    yield block_records
                current_block_id = block_id

                if record_set_num == "1":
                    block_set_1 = [(record_id, unlabeled_data_1[record_id])]
                    block_set_2 = []
                elif record_set_num == "2":
                    block_set_1 = []
                    block_set_2 = [(record_id, unlabeled_data_2[record_id])]
                else:
                    raise ValueError("record_set_num should only be 1 or 2, but got {}".format(record_set_num))
        
        block_records = (block_set_1, block_set_2)
        yield block_records

def get_blocked_pairs(deduper_or_linker, blocked_data):
    if isinstance(deduper_or_linker, dedupe.api.DedupeMatching):
        pairs = (combinations(sorted(block), 2) for block in blocked_data)
    elif isinstance(deduper_or_linker, dedupe.api.RecordLinkMatching):
        pairs = (product(base, target) for base, target in blocked_data)
    else:
        raise ValueError("Passed not of class DedupeMatching or of RecordLinkMatching!")
    return pairs

def count_blocked_pairs(deduper_or_linker, blocked_data):
    candidate_records = itertools.chain.from_iterable(get_blocked_pairs(deduper_or_linker, blocked_data))
    i = 0
    for _ in candidate_records:
        i += 1
    return i

def write_training_set_from_pairs(labeled_pair_ids_df, labeled_data_filepath, unlabeled_data, unlabeled_data_2 = None):
    # create a labeled training set directly for dedupe's consumption
    labeled_data_train = {"distinct":[], "match":[]}

    for _, row in labeled_pair_ids_df.iterrows():
        rec_id_1 = row["record id 1"]
        rec_id_2 = row["record id 2"]

        rec_data_1 = unlabeled_data[rec_id_1]
        if unlabeled_data_2 is None:
            rec_data_2 = unlabeled_data[rec_id_2]
        else:
            rec_data_2 = unlabeled_data_2[rec_id_2]

        label = row["label"]

        data_entry = {
            "__class__":"tuple",
            "__value__":[rec_data_1, rec_data_2]
        }
        labeled_data_train[label].append(data_entry)

    with open(labeled_data_filepath, "w") as json_file:
        simplejson.dump(labeled_data_train, 
                      json_file,
                      default=dedupe.serializer._to_json,
                      tuple_as_array=False,
                      ensure_ascii=True)

def get_deduped_data_for_rl(task_name, saved_files_path):
    # gets deduped dataset from respective deduping for rl
    
    dataset_name = task_name.split("-")[1]
    dataset_1_name, dataset_2_name = dataset_name.split("_")
    dedup_task_1 = "dedup-{}".format(dataset_1_name)
    dedup_task_2 = "dedup-{}".format(dataset_2_name)
    
    # get all filepaths
    unlabeled_data_1_filepath, unlabeled_data_2_filepath = dm_file_checker.get_proper_unlabeled_data_filepath(task_name, saved_files_path)
    numeric_fields_1, numeric_fields_2 = dm_file_checker.get_dataset_info(task_name, "numeric_fields", saved_files_path)
    print("Numeric fields 1 are {}".format(numeric_fields_1))
    print("Numeric fields 2 are {}".format(numeric_fields_2))
    
    canonicals_1_filepath = dm_file_checker.get_filepath(dedup_task_1, "cluster_canonical", saved_files_path)
    canonicals_2_filepath = dm_file_checker.get_filepath(dedup_task_2, "cluster_canonical", saved_files_path)
    
    mapped_records_1_filepath = dm_file_checker.get_filepath(dedup_task_1, "mapped_records", saved_files_path)
    mapped_records_2_filepath = dm_file_checker.get_filepath(dedup_task_2, "mapped_records", saved_files_path)
    
    # read in data from filepaths
    unlabeled_data_1 = read_unlabeled_data_json(unlabeled_data_1_filepath, empty_str_to_none = False, 
                                                numeric_fields = numeric_fields_1)
    unlabeled_data_2 = read_unlabeled_data_json(unlabeled_data_2_filepath, empty_str_to_none = False, 
                                                numeric_fields = numeric_fields_2)
    
    canonicals_1_df = pd.read_csv(canonicals_1_filepath, keep_default_na = False, low_memory = False)
    canonicals_2_df = pd.read_csv(canonicals_2_filepath, keep_default_na = False, low_memory = False)
    
    mapped_records_1_df = pd.read_csv(mapped_records_1_filepath, keep_default_na = False)
    mapped_records_2_df = pd.read_csv(mapped_records_2_filepath, keep_default_na = False)
    
    # get deduped data in dictionary form
    deduped_data_1 = get_deduped_data(mapped_records_1_df, canonicals_1_df, unlabeled_data_1, none_to_empty_str = False)
    
    deduped_data_2 = get_deduped_data(mapped_records_2_df, canonicals_2_df, unlabeled_data_2, none_to_empty_str = False)

    if numeric_fields_1 is not None:
        for col in numeric_fields_1:
            deduped_data_1[col] = deduped_data_1[col].apply(lambda x: x if x == "" else float(x))
    if numeric_fields_2 is not None:
        for col in numeric_fields_2:
            deduped_data_2[col] = deduped_data_2[col].apply(lambda x: x if x == "" else float(x))

    for col in deduped_data_1.columns:
        empty_str_bool = (deduped_data_1[col] == "")
        print("in deduped data 1, converting {} empty string values of column {} to None".format(empty_str_bool.sum(), col))
        deduped_data_1.loc[empty_str_bool,col] = None

    for col in deduped_data_2.columns:
        empty_str_bool = (deduped_data_2[col] == "")
        print("in deduped data 2, converting {} empty string values of column {} to None".format(empty_str_bool.sum(), col))
        deduped_data_2.loc[empty_str_bool,col] = None

    # converting NaNs of numeric columns (NaNs introduced because of the previous line) to None
    if numeric_fields_1 is not None:
        for col in numeric_fields_1:
            not_nan_bool = deduped_data_1[col].notnull()
            print("in deduped data 1, converting {} NaN values of {} to None".format((~not_nan_bool).sum(), col))
            deduped_data_1[col] = deduped_data_1[col].where((not_nan_bool), None)

    if numeric_fields_2 is not None:
        for col in numeric_fields_2:
            not_nan_bool = deduped_data_2[col].notnull()
            print("in deduped data 2, converting {} NaN values of {} to None".format((~not_nan_bool).sum(), col))
            deduped_data_2[col] = deduped_data_2[col].where((not_nan_bool), None)


    deduped_data_1 = deduped_data_1.to_dict(orient = "index")
    deduped_data_2 = deduped_data_2.to_dict(orient = "index")
    
    return deduped_data_1, deduped_data_2

# function to make sure the all record ids are prepended with the name of the dataset
def verify_rec_id_format(record_id, data_name):
    if pd.isnull(record_id):
        is_ok = True
    else:
        is_ok = (record_id.split("-")[0] == data_name)
    return is_ok

# function to return all results from all record linkage results
def get_all_rl_results(rl_task_names, saved_files_path):
    dupe_records = pd.DataFrame(columns = ["record id 1", "record id 2", "confidence score"])
    all_records = set()

    # iterate over each rl mapped file
    for rl_task in rl_task_names:
        data_name_1, data_name_2 = rl_task.split("-")[1].split("_")

        mapped_records_filepath = dm_file_checker.get_filepath(rl_task, "mapped_records", saved_files_path)
        print("Getting mapped record links from {}".format(rl_task))
        mapped_records_df = pd.read_csv(mapped_records_filepath)

        # make sure all record ids are prepended with the name of the dataset
        ok_records_1 = mapped_records_df["record id 1"].apply(lambda x: verify_rec_id_format(x, data_name_1)).all()
        ok_records_2 = mapped_records_df["record id 2"].apply(lambda x: verify_rec_id_format(x, data_name_2)).all()
        assert (ok_records_1 and ok_records_2), "Record ids aren't prepended with the dataset name!"

        append_dupe_records = mapped_records_df.loc[mapped_records_df["link type"] == "dup",\
                                             ["record id 1", "record id 2","confidence score"]]
        dupe_records = dupe_records.append(append_dupe_records, ignore_index = True)

        append_all_records = mapped_records_df.loc[:,["record id 1","record id 2"]]
        append_all_records = append_all_records["record id 1"].dropna().unique().tolist() \
                            + append_all_records["record id 2"].dropna().unique().tolist()
        append_all_records = set(append_all_records)
        all_records.update(append_all_records)
        
    all_records = list(all_records)
    pairs = dupe_records.loc[:,["record id 1", "record id 2"]]\
                        .apply(lambda row: (row["record id 1"], row["record id 2"]), axis = 1)\
                        .tolist()
    n_pairs = len(pairs)
    
    id_type = (str, 265)
    pairs = np.array(pairs, dtype = id_type)
    
    scores = dupe_records.loc[:,["confidence score"]].to_numpy(dtype = float).reshape(-1)

    
    dtype = np.dtype([("pairs", id_type, 2),
                     ("score", "f4", 1)])
    
    temp_file, file_path = tempfile.mkstemp()
    os.close(temp_file)
    
    scored_pairs = np.memmap(file_path, 
                           shape = n_pairs,
                           dtype = dtype)
    
    scored_pairs["pairs"] = pairs
    scored_pairs["score"] = scores
        
    return scored_pairs, all_records

def get_fusion_probs_and_threshold(scored_pairs, recall_weight = 1):
    probs = scored_pairs['score']

    probs = probs.copy()
    probs.sort()
    probs = probs[::-1]

    expected_dupes = np.cumsum(probs)

    recall = expected_dupes / expected_dupes[-1]
    precision = expected_dupes / np.arange(1, len(expected_dupes) + 1)

    score = recall * precision / (recall + recall_weight ** 2 * precision)

    i = np.argmax(score)

    print('Maximum expected recall and precision')
    print('recall: {:.2f}%'.format(recall[i]*100))
    print('precision: {:.2f}%'.format(precision[i]*100))
    print('With threshold: {:.2f}%'.format(probs[i]*100))

    return probs, probs[i]

def map_cluster_fusion_ids(scored_pairs, all_records, threshold):
    
    clustered_dupes = dedupe_cluster(scored_pairs, threshold)
    
    mapped_records = []
    record_ids_in_clusters = []

    # assign cluster ids to record ids
    i = 0
    print("Mapping cluster ids...")
    for cluster in tqdm(clustered_dupes):
        i += 1
        cluster_id = "fs-{}".format(i)
        id_set, scores = cluster

        for record_id, score in zip(id_set, scores):
            record_dict = {
                "record id": record_id,
                "cluster id": cluster_id,
                "confidence score": score,
                "cluster type":'link'
            }

            mapped_records.append(record_dict)
            
            record_ids_in_clusters.append(record_id)

    record_ids_in_clusters = set(record_ids_in_clusters)
    solo_ids = [key for key in all_records if key not in record_ids_in_clusters]

    # assign solo ids to record ids
    print("Mapping solo record ids...")
    for record_id in tqdm(solo_ids):
        i += 1
        cluster_id = "fs-{}".format(i)
        record_dict = {
            "record id":record_id,
            "cluster id":cluster_id,
            "confidence score":None,
            "cluster type":'solo'
        }
        
        mapped_records.append(record_dict)

    mapped_records = pd.DataFrame(mapped_records)

    return mapped_records

def get_all_dedup_results(rl_task_names, saved_files_path, remove_data_name_prefix = True):
    all_dedup_mapped_records = pd.DataFrame()
    dedup_datasets = set()

    for rl_task in rl_task_names:
        data_name_1, data_name_2 = rl_task.split("-")[1].split("_")

        for data_name in [data_name_1, data_name_2]:
            dedup_task = "dedup-{}".format(data_name)
            mapped_records_filepath = dm_file_checker.get_filepath(dedup_task, "mapped_records", saved_files_path)

            # replace IDs only of datasets that have undergone deduplication
            if os.path.exists(mapped_records_filepath) & (data_name not in dedup_datasets):
                dedup_datasets.add(data_name)

                dedup_mapped_records = pd.read_csv(mapped_records_filepath)
                dedup_mapped_records = dedup_mapped_records.rename(columns = {"confidence score":"dedup confidence score",
                                                                       "cluster type":"dedup cluster type"})
                
                if remove_data_name_prefix:
                    dedup_mapped_records["record id"] = dedup_mapped_records["record id"]\
                                                            .apply(lambda x: x.replace("{}-".format(data_name), ""))
                    
                all_dedup_mapped_records = all_dedup_mapped_records.append(dedup_mapped_records, ignore_index = True)
                
    return all_dedup_mapped_records

def check_block_sizes(blocks):
    block_sizes = []
    for block in blocks:
        block_size = len(block)
        block_sizes.append(block_size)
    block_sizes = sorted(block_sizes, reverse = True)
    
    print("Sizes of top 10 biggest blocks are: {}".format(block_sizes[:10]))
    record_pair_contributions = [int(size*(size-1)/2) for size in block_sizes[:10]]
    print("Record pair contributions from top 10 biggest blocks are : {}".format(record_pair_contributions))