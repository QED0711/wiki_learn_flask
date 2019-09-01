import flask
from flask import Flask, request
from flask_cors import CORS

import sys
sys.path.append("./utils")

import pickle

###########
## UTILS ##
###########

import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import requests
import re
import threading

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


with open("../wiki_learn/models/rf_classifier_v1.pkl", "rb") as model:
    rf_classifier = pickle.load(model)

application = Flask(__name__)
CORS(application)

@application.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Hello World"


    if request.method == "POST":
        entry = request.json['entry']
        gc = GraphCreator(entry)

        if len(gc.graph.nodes) > 500:
            return "Too Large"
        
        gc.expand_network_threaded(threads=5, chunk_size=1)
        gc.redraw_redirects()
        gc.update_edge_weights()
        features_df = gc.get_features_df(rank=False)
        gc.rank_similarity()
        scaled_feature_df = gc.scale_features_df(scaler=MinMaxScaler, copy=True) # Makes a copy of the df
        sorted_scaled = scaled_feature_df.sort_values("similarity_rank", ascending=False).reset_index().drop("index", axis=1)

        # drop the entry node row from recommendations
        # limit to first 100 recommendations
        sorted_scaled = sorted_scaled[sorted_scaled.node != gc.entry][0:100]

        # format df for predictions
        X = sorted_scaled.drop(["node", "similarity_rank"], axis=1)

        y_preds = rf_classifier.predict_proba(X)

        classes = list(rf_classifier.classes_)

        sorted_scaled['label'] = list(map(lambda x: list(x), y_preds))

        nodes = sorted_scaled[["node", "label", "similarity_rank"]].to_dict(orient="index")

        results = [{"entry": entry}, {"classes": classes}, nodes]

        return flask.jsonify(results)


if __name__ == "__main__":
    application.run(host='0.0.0.0')