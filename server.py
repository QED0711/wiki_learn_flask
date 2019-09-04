
import flask
from flask import Flask, request, Response
from flask_cors import CORS

import sys
sys.path.append("./utils")

from RecommenderPipeline import Recommender
from GraphAPI import GraphCreator
from graph_helpers import create_dispersion_df, sort_dict_values, format_categories, compare_categories, rank_order, similarity_rank
from WikiMultiQuery import wiki_multi_query
from wiki_scrapper import WikiScrapper
from url_utils import *

# import warnings
import networkx as nx
from sklearn.preprocessing import normalize, StandardScaler, Normalizer, RobustScaler, MinMaxScaler, MaxAbsScaler
from collections import Counter
# from functools import reduce
import pandas as pd
import numpy as np
import concurrent.futures
import threading
import re
import requests
# import pdb
# import seaborn as sns
# import matplotlib.pyplot as plt
import pickle
# import time


###########
## UTILS ##
###########


# warnings.filterwarnings("ignore")


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
        def generator(entry):
            # yield "something here"

            # yield entry

            gc = GraphCreator(entry)
            
            # yield f"Layer 1 nodes: {len(gc.graph.nodes)}\n"
            yield str({"layer 1 nodes":len(gc.graph.nodes)}) + "\n"

            # if len(gc.graph.nodes) > 500:
            #     return "Too Large"

            rec = Recommender(gc, threads=20, chunk_size=1)
            yield "Recommender Initialized\n"

            rec.fit(scaler=Normalizer)
            yield "recommend fit\n"
            
            rec.predict(rf_v2_classifier)
            yield "predictions complete\n"

            results = rec.format_results(decision_threshold=0.5)
            yield "Results formatted\n"

            results['classes'] = list(rec.classes)
            yield "Classes added\n"

            yield str(results)
        
        return Response(generator(entry))


@application.route("/test", methods=["POST"])
def test():
    if request.method == "POST":
        def generator():
            yield "one"
            print("sent one")
            time.sleep(2)
            yield "two"
            print("sent two")
            time.sleep(3)
            yield "three"
            print("sent three")
        return Response(generator())


if __name__ == "__main__":
    application.run(host='0.0.0.0')
