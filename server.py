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
from RecommenderPipeline import Recommender

with open("./models/rf_classifier_v2_normalized.pkl", "rb") as model:
    rf_v2_classifier = pickle.load(model)

application = Flask(__name__)
CORS(application)

@application.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Hello World"


    if request.method == "POST":
        entry = request.json['entry']

        print("Entry:", entry)

        gc = GraphCreator(entry)

        if len(gc.graph.nodes) > 500:
            return "Too Large"
        
        gc = GraphCreator(entry)
        rec = Recommender(gc, threads=20, chunk_size=1)
        
        rec.fit(scaler=Normalizer)
        rec.predict(rf_v2_classifier)
        results = rec.format_results(decision_threshold=0.5)        
        results['classes'] = list(rec.classes)

        return flask.jsonify(results)


if __name__ == "__main__":
    application.run(host='0.0.0.0')