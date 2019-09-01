import sys
sys.path.append("../")

import pdb

import requests
import pandas as pd
import re

import signal
import time

import pickle

import warnings
warnings.filterwarnings("ignore")



def has_key(key, resp, pageid):
    return bool(resp['query']["pages"][pageid].get(key))

def matches(lst1, lst2):
    return [x for x in lst1 if x in lst2]
    
    
def get_article_info(title):
    params = {
        "action": "query",
        "format": "json",

        "titles": title,

        "prop": "extracts|info|links|linkshere|categories",
        
        # extracts
        "exintro": True,
        "explaintext": True,
        "exsectionformat": "plain",
        
        # links
        "pllimit": "max",
        "plnamespace": 0,
        
        # linkshere
        "lhlimit": "max",
        "lhnamespace": 0,
        "lhshow": "!redirect",
        
        # categories
        "cllimit": "max",
    }

    extract = []
    links = []
    linkshere = []
    categories = []

    def query_info(title, params):

        resp = requests.get(
            url="https://en.wikipedia.org/w/api.php",
            params=params).json()

        pageid = list(resp["query"]['pages'].keys())[0]
        
        if has_key("extract", resp, pageid):
            extract.append(resp['query']["pages"][pageid]['extract'])
        
        if has_key("links", resp, pageid):
            for link in resp['query']["pages"][pageid]["links"]:
                links.append(link["title"])
        
        if has_key("linkshere", resp, pageid):
            for lh in resp['query']["pages"][pageid]["linkshere"]:
                linkshere.append(lh["title"])
        
        if has_key("categories", resp, pageid):
            for cat in resp['query']["pages"][pageid]["categories"]:
                if not bool(re.findall(r"(articles)|(uses)|(commons)", cat["title"], re.I)):
                    categories.append(cat["title"])
        
        if resp.get('continue'):
            params.update(resp.get("continue"))
            query_info(title, params)

    query_info(title, params)
    
    return extract, links, linkshere, categories




if __name__ == "__main__":
    parent_extract, parent_links, parent_linkshere, parent_categories = get_article_info("Random forest")
    print(parent_links)
