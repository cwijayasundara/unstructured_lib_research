# Warning control
import warnings

warnings.filterwarnings('ignore')

import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

import json
from IPython.display import JSON

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import dict_to_elements
import chromadb

from Utils import Utils

utils = Utils()

DLAI_API_KEY = utils.get_dlai_api_key()
DLAI_API_URL = utils.get_dlai_url()

s = UnstructuredClient(
    api_key_auth=DLAI_API_KEY,
    server_url=DLAI_API_URL,
)

# Example Document: Winter Sports in Switzerland EPUB

from IPython.display import Image

Image(filename='images/winter-sports-cover.png', height=400, width=400)
Image(filename="images/winter-sports-toc.png", height=400, width=400)

filename = "example_files/winter-sports.epub"

with open(filename, "rb") as f:
    files = shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(files=files)

try:
    resp = s.general.partition(req)
except SDKError as e:
    print(e)

json_response = json.dumps(resp.elements[0:3], indent=2)
print("The response is", json_response)

# Find elements associated with chapters
[x for x in resp.elements if x['type'] == 'Title' and 'hockey' in x['text'].lower()]

chapters = [
    "THE SUN-SEEKER",
    "RINKS AND SKATERS",
    "TEES AND CRAMPITS",
    "ICE-HOCKEY",
    "SKI-ING",
    "NOTES ON WINTER RESORTS",
    "FOR PARENTS AND GUARDIANS",
]

chapter_ids = {}
for element in resp.elements:
    for chapter in chapters:
        if element["text"] == chapter and element["type"] == "Title":
            chapter_ids[element["element_id"]] = chapter
            break

print("Chapter IDs:", chapter_ids)

chapter_to_id = {v: k for k, v in chapter_ids.items()}
[x for x in resp.elements if x["metadata"].get("parent_id") == chapter_to_id["ICE-HOCKEY"]][0]

# Load documents into a vector db
client = chromadb.PersistentClient(path="chroma_tmp", settings=chromadb.Settings(allow_reset=True))
client.reset()

collection = client.create_collection(
    name="winter_sports",
    metadata={"hnsw:space": "cosine"}
)

for element in resp.elements:
    parent_id = element["metadata"].get("parent_id")
    chapter = chapter_ids.get(parent_id, "")
    collection.add(
        documents=[element["text"]],
        ids=[element["element_id"]],
        metadatas=[{"chapter": chapter}]
    )

# See the elements in Vector DB
results = collection.peek()
print(results["documents"])

# Perform a hybrid search with metadata
result = collection.query(
    query_texts=["How many players are on a team?"],
    n_results=2,
    where={"chapter": "ICE-HOCKEY"},
)
print(json.dumps(result, indent=2))

# Chunking Content
elements = dict_to_elements(resp.elements)
chunks = chunk_by_title(
    elements,
    combine_text_under_n_chars=100,
    max_characters=3000,
)

response = json.dumps(chunks[0].to_dict(), indent=2)
print("The response is", response)
