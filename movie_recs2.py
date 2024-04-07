import pymongo
import openai
import os
"""

    OPENAI through embedding generation not working til now :(

"""

openai.api_key = os.getenv("OPENAI_API_KEY")

client = pymongo.MongoClient(os.getenv("PYMONGO_URL"))
db = client.sample_mflix
collection = db.embedded_movies

def generate_embedding(text: str) -> list[float]:
    response = openai.completions.create(
        model="text-embedding-ada-002", 
        prompt=text,
        max_tokens=0,
        n=1 
    )
    return response['data'][0]['embedding']

query = "imaginary characters from outer space at war"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch",
      }}
])

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')