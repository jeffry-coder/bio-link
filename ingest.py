import os
from dotenv import load_dotenv
from tqdm import tqdm
from uuid import uuid4

from arango import ArangoClient
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = vector_client.Index("synthea-desc")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

graph_client = ArangoClient(hosts=os.getenv("ARANGO_HOST")).db(
    username=os.getenv("ARANGO_USERNAME"),
    password=os.getenv("ARANGO_PASSWORD"),
    verify=True,
)

collections = [
    "allergies",
    "careplans",
    "conditions",
    "devices",
    "encounters",
    "immunizations",
    "medications",
    "observations",
    "procedures",
    "supplies",
]

documents = []

for col in tqdm(collections, desc="Processing collections"):
    query = f"""
    LET unique_descriptions = (
      FOR doc IN {col}
        COLLECT description = doc.DESCRIPTION INTO group
        RETURN {{
          "description": description,
          "_key": FIRST(group[*].doc._key)
        }}
    )
    FOR item IN unique_descriptions
      RETURN item
    """
    cursor = graph_client.aql.execute(query)
    for doc in cursor:
        documents.append(
            Document(
                page_content=doc["description"],
                metadata={"_key": doc["_key"], "collection": col},
            )
        )


uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents, ids=uuids)

results = vector_store.similarity_search(
    "chronic respiratory disease",
    k=3,
    filter={"collection": "conditions"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
