
import threading
import signal
from queue import Queue

from WikiMultiQuery import wiki_multi_query


class ThreadedQuery:
    """
    A multithreaded implementation of the 'wiki_multi_query' module.
    """
    def __init__(self, nodes):
        self.nodes = nodes

    def execute(self):
        article_queue = Queue()

        

