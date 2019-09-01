import requests


# Helper Functions

def query_wiki_api(params):
    url="https://en.wikipedia.org/w/api.php"
    return requests.get(url, params).json()

def get_pages(resp):
    return list(resp['query']["pages"].keys())

def page_dicts(resp):
    pages = get_pages(resp)
    return [{"pageid": page} for page in pages]

def add_title(resp, pages):
    for article in pages:
        try:
            pageid = article['pageid']
            article['title'] = resp['query']['pages'][pageid]['title']
        except:
            continue
    return pages

def add_redirects(resp, pages):
    for article in pages:
        try:
            pageid = article['pageid']
            redirects = [redirect['title'] for redirect in resp['query']['pages'][pageid]['redirects']]
            if "redirects" in article:
                article['redirects'] += redirects
            else:
                article['redirects'] = redirects
        except:
            continue
    
    return pages

def add_links(resp, pages):
    for article in pages:
        try:
            pageid = article['pageid']
            links = [link['title'] for link in resp['query']['pages'][pageid]['links']]
            if "links" in article:
                article['links'] += links
            else:
                article['links'] = links
        except:
            continue
            
    return pages

def add_linkshere(resp, pages):
    for article in pages:
        try:
            pageid = article['pageid']
            links = [link['title'] for link in resp['query']['pages'][pageid]['linkshere']]
            if "linkshere" in article:
                article['linkshere'] += links
            else:
                article['linkshere'] = links
        except:
            continue
            
    return pages

def add_categories(resp, pages):
    for article in pages:
        try:
            pageid = article['pageid']
            categories = [cat['title'] for cat in resp['query']['pages'][pageid]['categories']]
            if "categories" in article:
                article['categories'] += categories
            else:
                article['categories'] = categories
        except:
            continue
            
    return pages

def update_continue(resp, params):
    if resp.get("continue"):
        # remove any previous continue strings
        keys = list(params.keys())
        for key in keys:
            if "continue" in key:
                del params[key]
                
        # update with new params
        params.update(resp.get("continue"))        
        return params
    else:
        return False

######################
## Wiki Multi Query ##
######################

def wiki_multi_query(articles, params=None, pages=None, max_requests=20, request_count=0):
    """
    Accepts multiple article titles and returns link, linkhere, and category information for each.

    Input:
    ------
    articles (required) 
    A list of article titles

    params (optional)
    A dictionary of parameters. Must include at least, {"action": "query", "format": "json"}. 
    If no params are set, they will be automatically set.

    pages (None)
    This value will be set automatically upon recursive calls as necessary. 

    max_requests (default: 20)
    The maximum number of requests allowed for a set of articles.

    request_count (default: 0)
    the number of requests made for a set of articles. 
    Note: do not set this manually. 

    Returns:
    --------
    A list of dictionaries containing each article's information.
    """

    if not params:
        params = {
            "action": "query",
            "format": "json",

            "prop": "redirects|links|linkshere|categories",

            # redirects
            "rdnamespace": 0,
            "rdlimit": "max",

            # links
            "pllimit": "max",
            "plnamespace": 0,

            # linkshere
            "lhlimit": "max",
            "lhnamespace": 0,
            "lhshow": "!redirect",

            # categories
            "cllimit": "max",

            # automatic redirect
            "redirects": 1
        }



    params['titles'] = "|".join(articles)

    resp = query_wiki_api(params)

    request_count += 1

    if not pages:
        pages = page_dicts(resp)

    pages = add_title(resp, pages)
    pages = add_redirects(resp, pages)
    pages = add_links(resp, pages)
    pages = add_linkshere(resp, pages)
    pages = add_categories(resp, pages)

    # will return an updated params with continue statements OR False
    params = update_continue(resp, params)

    # if the max requests limit is exceded
    if params and request_count >= max_requests:
        return pages

    # if params still is truthy, then it was updated with a continue
    # start the process again on the continued params
    if params:
        return wiki_multi_query(articles, params, pages, max_requests, request_count)

    return pages



if __name__ == "__main__":
    print(wiki_multi_query(["Random forest", "Decision tree", "Machine learning", "Sonata"])[3]['redirects'])





