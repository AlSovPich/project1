import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
from urllib.parse import urljoin

class RAGModule:
    def __init__(self, chromadb_client=None):
        self.chromadb_client = chromadb_client or chromadb.Client()
        self.collection = self.chromadb_client.create_collection("my_docs")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.visited_urls = set()

    def extract_info_from_url(self, url):
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            extracted_info = []
            for element in soup.find_all(['h1', 'h2', 'p', 'li']):
                text = element.get_text(strip=True)
                if text:
                    extracted_info.append(text)

            embeddings = self.model.encode(extracted_info)
            ids = [f"{url}_{i}" for i in range(len(extracted_info))]
            self.collection.add(
                documents=extracted_info,
                embeddings=embeddings,
                ids=ids
            )

            for link in soup.find_all('a', class_="reference internal"):
                link_url = link['href']
                if link_url.startswith('/'):
                    link_url = urljoin(url, link_url)
                self.extract_info_from_url(link_url)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {url}, Error: {e}")

    def query_chromadb(self, query_text):
        query_embedding = self.model.encode([query_text])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=5)
        return results['documents'][0]