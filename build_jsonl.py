import os
import json
from Bio import Entrez

# Configure NCBI access
Entrez.email = os.getenv("NCBI_EMAIL", "your_email@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY")

# 1. Query PubMed for meningitis case reports and reviews
query = 'meningitis AND (case report[pt] OR review[pt])'
search_handle = Entrez.esearch(db="pubmed", term=query, retmax=50)
search_record = Entrez.read(search_handle)
id_list = search_record["IdList"]

# 2. Fetch full XML records for those IDs
fetch_handle = Entrez.efetch(
    db="pubmed",
    id=",".join(id_list),
    rettype="xml",
    retmode="xml"
)
records = Entrez.read(fetch_handle)["PubmedArticle"]

# 3. Build JSONL records
output = []
for art in records:
    pmid = art["MedlineCitation"]["PMID"]
    article = art["MedlineCitation"]["Article"]
    title = article.get("ArticleTitle", "").strip()
    abstract_nodes = article.get("Abstract", {}).get("AbstractText", [])
    abstract = " ".join(str(a) for a in abstract_nodes).strip()
    if not abstract:
        continue

    output.append({
        "prompt": f"Title: {title}\nAbstract: {abstract}\n\nRetrieve similar cases:",
        "metadata": {
            "pmid": pmid,
            "source": "pubmed"
        }
    })

# 4. Write to data.jsonl
with open("data.jsonl", "w") as f:
    for rec in output:
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {len(output)} case-report and review records to data.jsonl")