import requests
from bs4 import BeautifulSoup

from url_utils import parse_url, get_title
from get_related_links import get_see_also_links

class WikiScrapper:
    
    def __init__(self, url):
        self.url = url
        
        self.title = None
        
        self.intro_links = []
        self.parsed_links = []
        self.intro_link_titles = []

        see_also_links = get_see_also_links(self.url)
        if see_also_links:
            self.see_also_link_titles = see_also_links["links"]
        else:
            self.see_also_link_titles = []
        
    
    def get_primary_links(self, include_see_also=True):
        """
        Returns scrapped links from the intro text and see also section. 
        
        Optional: If include_see_also is set to False, the see_also_link_titles will not
        be included in the result. This may be desirable if you want to evaluate the performance
        of the recomendation metric by seeing how many of the see also links it was able to pull out
        without prior knowledge of their existance. 
        """
        if include_see_also:
            return list(set(self.intro_link_titles + self.see_also_link_titles))
        else: 
            return list(set(self.intro_link_titles))

    def _get_article(self):
        resp = requests.get(self.url)
        soup = BeautifulSoup(resp.content)
        self._set_title(soup)
        return soup
        
    def _set_title(self, soup):
        self.title = soup.find_all("h1")[0].text

    
    def _get_intro_links(self):
        soup = self._get_article().find_all(class_="mw-parser-output")[0]
        
        children = list(soup.children)
        
        started_intro_text = False
        
        for child in children:
            if child.name == "p" and child.has_attr("class") == False:
                started_intro_text = True
                self.intro_links += child.find_all("a")
            if child.name != "p" and started_intro_text:
                break
    
    def parse_intro_links(self):
        self._get_intro_links()
        
        for link in self.intro_links:
            current_href = link.get('href')
            if current_href.startswith("/wiki/") and not (":" in current_href):
                self.parsed_links.append(parse_url("https://en.wikipedia.org" + current_href))
                self.intro_link_titles.append(get_title(current_href))

        
        return self.parsed_links


if __name__ == "__main__":
    ws = WikiScrapper("https://en.wikipedia.org/wiki/Decision_tree")

    ws.parse_intro_links()

    print("Title:\t", ws.title)

    # print(ws.intro_link_titles)
    print(ws.get_primary_links())