
import sys
sys.path.append("./utils")


import pickle
import json
import requests
import re
import threading
import pymongo
import concurrent.futures
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import normalize, StandardScaler, Normalizer, RobustScaler, MinMaxScaler, MaxAbsScaler
import networkx as nx
from url_utils import *
from keys import *
from wiki_scrapper import WikiScrapper
from WikiMultiQuery import wiki_multi_query
from graph_helpers import create_dispersion_df, sort_dict_values, format_categories, compare_categories, rank_order, similarity_rank
from GraphAPI import GraphCreator
from RecommenderPipeline import Recommender
import flask
from flask import Flask, request, Response, stream_with_context
from flask_api import status
from flask_cors import CORS




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
        return "Server is connected"

    if request.method == "POST":

        entry = request.json['entry']

        def generator(entry):

            print(entry)
            gc = GraphCreator(entry)

            print(f"layer 1 nodes:{len(gc.graph.nodes)}")
            yield str({"layer 1 nodes": len(gc.graph.nodes)}) + "\n"

            # set max nodes limit
            if len(gc.graph.nodes) > 575:
                yield str({"error": "Network too large"})
                return

            rec = Recommender(gc, threads=50, chunk_size=1)
            print("Recommender System Initialized")
            yield "Recommender Initialized\n"

            # rec.fit(scaler=Normalizer)
            # yield "recommend fit\n"

            rec._expand_network()
            print("expanding\n")
            yield "expanding\n"

            rec._graph_cleanup()
            print("cleanup\n")
            yield "cleanup\n"

            rec._get_features()
            print("extracting features\n")
            yield "features\n"

            rec._calculate_similarity()
            print("calculating similarity index\n")
            yield "similarity\n"

            rec.scaled = rec._scale_features(Normalizer)
            print("scaling features\n")
            yield "scaled\n"

            rec.predict(rf_v2_classifier)
            yield "predictions complete\n"

            results = rec.format_results(decision_threshold=0.5)
            yield "Results formatted\n"

            results['classes'] = list(rec.classes)

            yield json.dumps(results)

        return Response(stream_with_context(generator(entry)), content_type="text/event-stream")


@application.route('/save', methods=["POST"])
def save():
    # get user labeled submitted data
    submission = request.json['submission']
    
    # setup for pymongo connection to mlab
    uri = f"mongodb://{mlab_api['username']}:{mlab_api['password']}@ds261277.mlab.com:61277/wiki_scrapper"    

    client = pymongo.MongoClient(uri)
    db = client.get_default_database()
    data_inserter = db["userLabeled"]

    data_inserter.insert_one(submission)

    client.close()

    # return success
    return "submission received", status.HTTP_201_CREATED

@application.route("/connect", methods=["POST"])
def connect():
    if request.method == "POST":
        return "Server Connected"


if __name__ == "__main__":
    application.run(host='0.0.0.0')
