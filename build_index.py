import os
import json
import time
from openai import OpenAI
import faiss
import numpy as np

# 0. Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Load prompts from data.jsonl
records = []
with open("data.jsonl", "r") as f:
    for line in f:
        records.append(json.loads(line))
prompts = [rec["prompt"] for rec in records]

# 2. Compute embeddings
embeddings = []
for prompt in prompts:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=prompt
    )
    embeddings.append(response.data[0].embedding)
    time.sleep(0.5)  # to respect rate limits

emb_array = np.array(embeddings, dtype="float32")

# 3. Build FAISS index
dim = emb_array.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(emb_array)
faiss.write_index(index, "index.faiss")

# 4. Save metadata for lookup
meta = [{"pmid": rec["metadata"]["pmid"], "prompt": rec["prompt"]} for rec in records]
with open("metadata.json", "w") as f:
    json.dump(meta, f)

print(f"Built FAISS index with {len(records)} vectors and saved to index.faiss")