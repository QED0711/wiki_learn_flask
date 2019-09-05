import sys
sys.path.append("../")

import pandas as pd
from copy import copy, deepcopy


from get_related_links import get_see_also_links
import time

class WikiScrapper:
    
    def __init__(self):
        self.data = []
    
    
    def traverse_from(self, url, max_depth=3, max_nodes=100):
        current = get_see_also_links(url)
        
        
        if current:
            queue = [link for link in current['links'] if self.valid_url(link)]
        else: 
            return

        self.data = [current]
        seen = {current["title"]: True}
        depth_count = 1

        while depth_count < max_depth and len(queue) > 0:
            queue_copy = queue.copy()
            
            for link in queue_copy:
                try:
                    current_url = queue.pop(0)
                    current = get_see_also_links(current_url)

                    if current:
                        if seen.get(current['title']):
                            continue

                        seen[current['title']] = True
                        
                        queue += current['links']
                        self.data.append(current)
                        if max_nodes and len(self.data) == max_nodes:
                            return self.data
                except:
                    continue
            
            depth_count += 1
        return self.data
        
    def valid_url(self, url):
        return len(url.split("//")) <= 2

    def to_dataframe(self):
        return pd.DataFrame(self.data)
    
    def to_csv(self, file_name):
        self.to_dataframe().to_csv(file_name, index=False)
    
    def add_ids(self):
        data = deepcopy(self.data)
        
        for article in data:
            article['_id'] = article['title']
        return data
    


if __name__ == "__main__":
    scrapper = WikiScrapper()
    scrapper.traverse_from("https://en.wikipedia.org/wiki/Decision_tree", max_depth=1, max_nodes=2)
    
    print(scrapper.data)