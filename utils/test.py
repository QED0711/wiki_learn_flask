import asyncio
import requests
import time

urls = [
    'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=categories&titles=Janelle%20Mon%C3%A1e',
    'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=categories&titles=Janelle%20Mon%C3%A1e'
]

def handle_request(url):
    def make_request(url):
        resp = requests.get(url)
        print(resp.text)

    return make_request(url)

async def main(urls):

    loop = asyncio.get_event_loop()

    for url in urls:
        print(url)
        future = loop.run_in_executor(None, requests.get, url)
        print(future)

        
        
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(urls))


