import pdb
import numpy as np
import pandas as pd
import networkx as nx

def sort_dict_values(dict, columns, sort_column, ascending=False):
    to_list = [(key, value) for key, value in dict.items()]
    return pd.DataFrame(to_list, columns=columns).sort_values(sort_column, ascending=ascending).reset_index().drop("index", axis=1)

def create_dispersion_df(G, central_node, node_list):
    print("MAKING DISPERSION")
    dispersion = [(central_node, node, nx.dispersion(G, central_node, node)) for node in node_list]
    return pd.DataFrame(dispersion, columns=["entry", "node", "dispersion"]).sort_values("dispersion", ascending=False).reset_index().drop("index", axis=1)

def format_categories(cat_list):
    cat_dict = {}
    for cat in cat_list:
        cat_dict[cat] = True    
    return cat_dict

def compare_categories(node1, node2, categories, starting_count=1):
    match_count = starting_count
    if node1 == node2:
        return 1
        
    try:
        for cat in categories[node1].keys():
            if categories[node2].get(cat):
                match_count += 1
        return match_count
    except:
        return starting_count

def rank_order(df, rank_column, ascending=False):
    """
    Given dataframe with a column of numerical values, order and rank those values.
    Allows for ties if a value is the same as the previous value. 
    """
    df = df.sort_values(rank_column, ascending=ascending).reset_index().drop('index', axis=1)
    rankings = [1]
    for i in range(1, df.shape[0]):
        # pdb.set_trace()
        if df[rank_column][i] == df[rank_column][i-1]: # if value is same as last val
            rankings.append(rankings[-1])
        else: # if value is different from last val
            rankings.append(rankings[-1] + 1)

    df[f"{rank_column}_ranked"] = rankings
    return df


def similarity_rank(row, **kwargs):
    """
    A helper method for use in `apply` to get the similarity rank from similarity bonuses and penalties
    Bonuses: category_matches_with_source, primary_link, shared_neighbors_with_entry_score
    Penalties: shortest_path_length_from_entry, shortest_path_length_to_entry
    """
    # degree_z_score = (row.degree - kwargs['degree_mean']) / kwargs['degree_std']

    bonus = row.category_matches_with_source + row.shared_neighbors_with_entry_score + row.primary_link + row.centrality + row.page_rank + row.adjusted_reciprocity
    # set penalty to the mean of the path lengths to/from the entry node
    penalty = np.mean([row.shortest_path_length_from_entry, row.shortest_path_length_to_entry])
    # if penalty is nan, just set it to the highest path length (the greatest penalty)
    if np.isnan(penalty):
        penalty = max(row.shortest_path_length_from_entry, row.shortest_path_length_to_entry) 

    # add any other penalty terms
    penalty += (row.degree / kwargs['degree_mean'])

    try:
        # similarity is penalized by longer paths
        sim_score = bonus / penalty   
        # if a path from the source does not exist, it is given a similarity score of 0
        return 0 if np.isnan(sim_score) else sim_score
    except:
        # in the case that we run into a divide by 0
        return 0