import pymongo
import os
import requests
import json

"""
    Embedding generation normally without openAI, using Huggingface API :
    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
"""

client = pymongo.MongoClient(os.getenv("PYMONGO_URL"))
db = client.sample_mflix
collection = db.movies

items = collection.find().limit(5)

with open("config.json", "r") as f:
    config = json.load(f)

hf_token = config.get("hf_token")
hf_api_url = config.get("hf_api_url")

embedding_url = hf_api_url
def generate_embedding(text: str) -> list[float]:
    response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}")

    return response.json()

"""
    input_text = "freeCodeCamp is awesome"
    print(generate_embedding(input_text))
"""

for doc in collection.find({'plot':{"$exists": True}}).limit(50):
    input_text = doc['plot']
    doc['plot_embedding_hf'] = generate_embedding(input_text)
    collection.replace_one({'_id': doc['_id']}, doc)

query = "Cartoon figures announce, via comic strip balloons, that they will move - and move they do, in a wildly exaggerated style."

results = collection.aggregate([
    {"$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticSearch",
        }
    }
])

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')


