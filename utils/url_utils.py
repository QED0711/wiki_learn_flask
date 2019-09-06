from urllib.parse import unquote

def parse_url(url):
    """
    returns a parsed url with utf-8 encoding
    """
    return unquote(url)

def get_title(url):
    """returns an article title from a url and replaces and "_" with spaces"""
    return parse_url(url.split("/")[-1]).replace("_", " ")


if __name__ == "__main__":
    print(parse_url("https://en.wikipedia.org/wiki/Elisabeth_R%C3%B6ckel"))
    print(get_title("https://en.wikipedia.org/wiki/Elisabeth_R%C3%B6ckel"))
    
    print(get_title("Decision tree"))