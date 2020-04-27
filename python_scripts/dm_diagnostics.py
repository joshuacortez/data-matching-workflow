import recordlinkage as rl
import pandas as pd
from sklearn.metrics import roc_auc_score
import json

def get_indexing_stats(true_links, candidate_pairs, n_total_pairs = None):
    if n_total_pairs is not None:
        reduction_ratio = 1 - (candidate_pairs.shape[0]/n_total_pairs)
    else:
        reduction_ratio = None
    pairs_completeness_ratio = true_links.isin(candidate_pairs).mean()
    pairs_quality = candidate_pairs.isin(true_links).mean()
    
    output = {"reduction_ratio":reduction_ratio,
              "pairs_completeness_ratio":pairs_completeness_ratio,
             "pairs_quality_ratio":pairs_quality}

    return output  

def diagnose_indexing(true_links, candidate_pairs, n_total_pairs = None):
    indexing_stats = get_indexing_stats(true_links, candidate_pairs, n_total_pairs)

    if indexing_stats["reduction_ratio"] is not None:
        print("Reduction Ratio")
        print("{:.2f}%".format(indexing_stats["reduction_ratio"]*100))

    print("Pairs Completeness Ratio")
    print("{:.2f}%".format(indexing_stats["pairs_completeness_ratio"]*100))

    print("Pairs Quality Ratio")
    print("{:.2f}%".format(indexing_stats["pairs_quality_ratio"]*100))
    
def diagnose_links(true_links, pred_links, total_n_links, similarity_df = None):
    confusion_mat = rl.confusion_matrix(true_links,pred_links,total = total_n_links)
    print("Confusion Matrix")
    print(confusion_mat)
    
    print("Comfusion Matrix (percentages)")
    print(confusion_mat*100/total_n_links)
    print("")
    
    matched_true_links = true_links.isin(pred_links)
    print("Recall Metrics:")
    print("Num of True Links: {:,}".format(len(true_links)))
    print("Num of True Links Matched: {:,}".format(matched_true_links.sum()))
    print("Num of True Links Unmatched: {:,}".format(len(matched_true_links) - matched_true_links.sum()))
    print("Percent of True Links Matched: {:.2f}%".format(matched_true_links.mean()*100))
    print("Percent of True Links Unmatched: {:.2f}%".format((1 - matched_true_links.mean())*100))
    print("")
    
    correct_predictions = pred_links.isin(true_links)
    print("Precision Metrics:")
    print("Num of Predicted Matches: {:,}".format(len(pred_links)))
    print("Num of Correct Predicted Matches: {:,}".format(correct_predictions.sum()))
    print("Num of Incorrect Predicted Matches: {:,}".format(len(correct_predictions) - correct_predictions.sum()))
    print("Percent of Predictions which are Correct: {:.2f}%".format(correct_predictions.mean()*100))
    print("Percent of Predictions which are Incorrect: {:.2f}%".format((1 - correct_predictions.mean())*100))
    print("")

    f1_score = rl.fscore(true_links, pred_links)
    print("F1 Score is {:.2f}%".format(f1_score*100))    

    if similarity_df is not None:
        is_true_link = similarity_df.index.isin(true_links)
    
        auc = roc_auc_score(y_true = is_true_link, y_score = similarity_df["similarity_score"])
        print("AUC of ROC of Similarity Scores is {:.2f}%".format(auc*100))

def diagnose_solos(true_solo_records, pred_solo_records):
    if len(true_solo_records) > 0:
    
        true_solo_caught = true_solo_records.isin(pred_solo_records)
        print("Recall Metrics:")
        print("Num of True Solo: {:,}".format(len(true_solo_records)))
        print("Num of True Solo Identified as Solo: {:,}".format(true_solo_caught.sum()))
        print("Num of True Solo Misidentified with Duplicate {:,}".format(len(true_solo_caught) - true_solo_caught.sum()))
        print("Percent of True Solo Identified as Solo: {:.2f}%".format(true_solo_caught.mean()*100))
        print("Percent of True Solo Misidentified with Duplicate: {:.2f}%".format((1 - true_solo_caught.mean())*100))
        print("")

        correct_predictions = pred_solo_records.isin(true_solo_records)
        print("Precision Metrics:")
        print("Num of Predicted Solo: {:,}".format(len(pred_solo_records)))
        print("Numb of Correct Predictions {:,}".format(correct_predictions.sum()))
        print("Num of Incorrect Predictions {:,}".format(len(correct_predictions) - correct_predictions.sum()))
        print("Percent of Predictions which are Correct {:.2f}%".format(correct_predictions.mean()*100))
        print("Percent of Predictions which are Incorrect {:.2f}%".format((1 - correct_predictions.mean())*100))
        
    else:
        print("No true solo records!")
        print("Number of Incorrectly Predicted Solo: {:,}".format(len(pred_solo_records)))

def diagnose_link_errors(true_links, pred_links, df, fields_list, 
                    similarity_df = None, error_type = "missed_true_links"):
    assert error_type in ["missed_true_links", "incorrect_predictions"], "select only from these error types"
    
    df = df.loc[:,fields_list].copy()
    
    if error_type == "missed_true_links":
        error_links = true_links[~true_links.isin(pred_links)]
    else:
        error_links = pred_links[~pred_links.isin(true_links)]
    
    df_left = df.loc[error_links.get_level_values(0),:].copy()
    df_left.columns = ["{}_1".format(col) for col in df_left.columns]
    df_left.index.name = "rec_id_1"
    df_left = df_left.reset_index()
    
    df_right = df.loc[error_links.get_level_values(1),:].copy()
    df_right.columns = ["{}_2".format(col) for col in df_right.columns]
    df_right.index.name = "rec_id_2"
    df_right = df_right.reset_index()
    
    output = pd.concat([df_left, df_right], axis = 1, sort = False)
    # sort columns
    output = output.sort_index(axis = 1)
    output = output.set_index(["rec_id_1", "rec_id_2"])

    if similarity_df is not None:
        assert "similarity_score" in similarity_df.columns
        output = pd.merge(left = output, right = similarity_df.loc[:,["similarity_score"]], 
                                left_index = True, right_index = True, how = "left")

    #output = output.drop_duplicates()
    print("Returning {}".format(error_type))
    return output

def diagnose_clusters(true_clusters, pred_clusters):
    matched_true_clusters = [cluster for cluster in true_clusters if cluster in pred_clusters]
    print("Recall Metrics:")
    print("Num of True Clusters: {:,}".format(len(true_clusters)))
    print("Num of True Clusters Matched: {:,}".format(len(matched_true_clusters)))
    print("Num of True Clusters Unmatched: {:,}".format(len(true_clusters) - len(matched_true_clusters)))
    print("Percent of True Clusters Matched: {:.2f}%".format(len(matched_true_clusters)*100/len(true_clusters)))
    print("Percent of True Clusters Unmatched: {:.2f}%".format(100 - len(matched_true_clusters)*100/len(true_clusters)))
    print("")
    
    correct_predictions = [cluster for cluster in pred_clusters if cluster in true_clusters]
    print("Precision Metrics:")
    print("Num of Predicted Clusters: {:,}".format(len(pred_clusters)))
    print("Num of Correct Predicted Matches: {:,}".format(len(correct_predictions)))
    print("Num of Incorrect Predicted Matches: {:,}".format(len(pred_clusters) - len(correct_predictions)))
    print("Percent of Predictions which are Correct: {:.2f}%".format(len(correct_predictions)*100/len(pred_clusters)))
    print("Percent of Predictions which are Incorrect: {:.2f}%".format(100 - len(correct_predictions)*100/len(pred_clusters)))
    print("")