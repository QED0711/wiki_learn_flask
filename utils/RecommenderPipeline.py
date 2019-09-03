import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import requests
import re
import threading
import pickle

import concurrent.futures

import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter

from sklearn.preprocessing import normalize, StandardScaler, Normalizer, RobustScaler, MinMaxScaler, MaxAbsScaler

import networkx as nx



import warnings
warnings.filterwarnings("ignore")

from url_utils import *
from wiki_scrapper import WikiScrapper
from WikiMultiQuery import wiki_multi_query
from graph_helpers import create_dispersion_df, sort_dict_values, format_categories, compare_categories, rank_order, similarity_rank

from GraphAPI import GraphCreator



# with open("../models/rf_classifier_v1.pkl", "rb") as model:
#     rf_classifier = pickle.load(model)

###############
# RECOMMENDER #
###############

class Recommender:

    def __init__(self, graph_creator, threads=20, chunk_size=1,):
        self.gc = graph_creator

        self.threads = threads
        self.chunk_size = chunk_size
        


    def fit(self, scaler=MinMaxScaler):
        self._expand_network()
        self._graph_cleanup()
        self._get_features()
        self._calculate_similarity()
        
        self.scaled = self._scale_features(scaler)

    def predict(self, model, size=100):
        
        self.classes = model.classes_

        # change these drop terms according to the best model
        X = self.scaled.drop([
            "node", 
            "similarity_rank",
            # "shortest_path_length_to_entry",
            # "primary_link",
            # "category_matches_with_source",
            # "shortest_path_length_from_entry",
        ], axis=1)[:size]
        preds = [list(x) for x in model.predict_proba(X)]

        result = self.scaled[["node", "similarity_rank"]][:size]
        result["label_proba"] = preds

        result = result[result.node != self.gc.entry]
        result_dict = result.to_dict(orient="index")
        self.predictions = [val for key, val in result_dict.items()]

    def format_results(self, decision_threshold=None):
        """
        Takes the calculated prediction probabilities and converts them to before/after labeles

        Input:
        ------
        decision_threshold (default: None, float between 0. and 1.)
        If set, will assign labeles based on on the decision threshold. If not set, will determine the optimum value where classes are most equal. Set to 0.5 if you want an unweighted decision.

        Returns:
        --------
        A final results dictionary with the entry, decision threshold (user set or calculated), and the class predictions
        """
        # if decision threshold is not set, calculate the aproximate optimum value where classes are
        # most equal
        if not decision_threshold:        
            thresholds = list(np.arange(0.25, 0.75, 0.01))
            differences = []
            for i, thresh in enumerate(thresholds):
                differences.append({"threshold": thresh, self.classes[0]: 0, self.classes[1]: 0})
                for node in self.predictions:
                    pred = node['label_proba'][0] > thresh
                    node['position'] = self.classes[0] if pred else self.classes[1]
                
                    if node['position'] == self.classes[0]:
                        differences[i][self.classes[0]] += 1
                    else:
                        differences[i][self.classes[1]] += 1
                differences[i]['difference'] = abs(differences[i][self.classes[0]] - differences[i][self.classes[1]])
        
            # set decision threshold based on optimum value 
            decision_threshold = sorted(differences, key=lambda x: x['difference'])[0]['threshold']
        
        for node in self.predictions:
            pred = node['label_proba'][0] > decision_threshold
            node['position'] = self.classes[0] if pred else self.classes[1]

        return {
            "entry": self.gc.entry,
            "decision_threshold": decision_threshold,
            "predictions": self.predictions
        }

        

    def _expand_network(self):
        self.gc.expand_network_threaded(threads=self.threads, chunk_size=self.chunk_size)

    def _graph_cleanup(self):
        self.gc.redraw_redirects()
        self.gc.update_edge_weights()

    def _get_features(self):
        self.gc.get_features_df(rank=False)

    def _calculate_similarity(self):
        self.gc.rank_similarity()

    def _scale_features(self, scaler=MinMaxScaler):
        return self.gc.scale_features_df(scaler=scaler, 
            copy=True).sort_values("similarity_rank", 
            ascending=False).reset_index().drop("index", axis=1)

    




if __name__ == "__main__":
    pass