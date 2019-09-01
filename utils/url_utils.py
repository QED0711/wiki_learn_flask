from urllib.parse import unquote

def parse_url(url):
    return unquote(url)

def get_title(url):
    return parse_url(url.split("/")[-1]).replace("_", " ")


if __name__ == "__main__":
    print(parse_url("https://en.wikipedia.org/wiki/Elisabeth_R%C3%B6ckel"))
    print(get_title("https://en.wikipedia.org/wiki/Elisabeth_R%C3%B6ckel"))
    
    print(get_title("Decision tree"))