import numpy as np
import json
from qdrant_client import QdrantClient
import pandas as pd

client = QdrantClient(url="https://b90cb3fe-db4f-4798-9724-b6d73a420db8.europe-west3-0.gcp.cloud.qdrant.io:6333/",
api_key="X8Fr3IQ4bPeu1KsseyE9erILiYAFafB9gAp865h6pqXjPw8g1BrN3w"
)

df = pd.read_csv('./organizations.csv', sep=',')

payload=df.to_dict('records')

vectors = np.load('./embeddings.npy')

client.upload_collection(
    collection_name='text-demo',
    vectors={
        "fast-bge-small-en" : vectors
    },
    payload=payload,
    ids=None,  # Vector ids will be assigned automatically
    batch_size=256  # How many vectors will be uploaded in a single request?
)