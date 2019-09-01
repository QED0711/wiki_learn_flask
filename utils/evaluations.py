
import pdb
import numpy as np
import pandas as pd

from math import ceil

def score_at_percentage(alpha, df, targets):
    segment = ceil(alpha * df.shape[0])
    segmented_df = df[0:segment]
    targets_seen = 0
    for i, row in segmented_df.iterrows():
        if row.node in targets:
            targets_seen += 1
    
    return targets_seen

def valid_targets(df, targets):
    validated = []
    for target in targets:
        if df[df.node == target].shape[0] > 0: # if target is in our dataframe
            validated.append(target)
    return validated


def evaluate_recommendations(df, on, targets):
    """
    Evaluates recommendation results based on how highly they rated the target values

    Input
    -----
    df (pandas dataframe): a dataframe containing the features and node names

    on (string): the column name that will be sorted on (how we rank our nodes/targets)

    targets (array-like): a list of known recommendations (from the "see also" section)


    return
    ------
    Returns a datafram report containing the score, the max score, and percentage points at which targets
    were found in the top recommendations.
    """
    # target = valid_targets(df, targets)
    sorted_df = df.sort_values(on, ascending=False).reset_index().drop("index", axis=1) 
    total_nodes = sorted_df.shape[0]

    max_target_index = 0

    for target in targets:
        try:
            target_val = sorted_df[sorted_df.node == target].index[0]
            if max_target_index < target_val:
               max_target_index = target_val 
        except:
            continue

    # Represents the percentage of rows we must go down before we have captured all targets
    # a higher number indicates that the targets are closer to the top of our recommendations
    score = 1 - (max_target_index / (total_nodes - len(targets)))
    max_score_possible = 1 - (len(targets) / total_nodes)
    report = pd.DataFrame([
        {"Metric Score": "score", on: score},
        {"Metric Score": "max score possible", on: max_score_possible},
        {"Metric Score": "difference", on: max_score_possible - score},
        {"Metric Score": "total targets", on: len(targets)},
        {"Metric Score": "% targets in top 1%", on: (score_at_percentage(0.01, sorted_df, targets) / len(targets))},
        {"Metric Score": "% targets in top 5%", on: (score_at_percentage(0.05, sorted_df, targets) / len(targets))},
        {"Metric Score": "% targets in top 10%", on: (score_at_percentage(0.10, sorted_df, targets) / len(targets))},
        {"Metric Score": "% targets in top 20%", on: (score_at_percentage(0.20, sorted_df, targets) / len(targets))},
    ]).set_index("Metric Score")

    return report.T

def evaluate_metrics(df, on, targets):
    validated_targets = valid_targets(df, targets)
    dfs = []
    for metric in on: 
        dfs.append(evaluate_recommendations(df, metric, validated_targets))

    return pd.concat(dfs)
