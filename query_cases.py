import os, sys, json, faiss
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
index = faiss.read_index("index.faiss")

# Load metadata and prompts
meta = json.load(open("metadata.json"))  # list of {"pmid": ..., "source": ...}
prompts = [json.loads(line) for line in open("data.jsonl", "r")]

def find_similar_pmids(profile: str, k: int = 5):
    resp = client.embeddings.create(model="text-embedding-ada-002", input=profile)
    qemb = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    _, indices = index.search(qemb, k)
    return [meta[i]["pmid"] for i in indices[0]], indices[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python3 query_cases.py "<patient description>"')
        sys.exit(1)
    profile = sys.argv[1]
    pmids, idxs = find_similar_pmids(profile, k=5)
    for rank, (pmid, i) in enumerate(zip(pmids, idxs), start=1):
        # extract title from the corresponding prompt record
        full_prompt = prompts[i]["prompt"]
        title_line = full_prompt.split("\n", 1)[0].replace("Title: ", "")
        print(f"{rank}. PMID {pmid}: {title_line}")