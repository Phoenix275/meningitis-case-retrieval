from Bio import Entrez
# replace with your email
Entrez.email = "your.email@school.edu"

# Gold PMIDs for Profile 1
gold = {"40622525","40696623","40656496","40622531","40705280"}

# Simple keyword query
query = "meningitis AND CSF neutrophils 1000"

# Search PubMed
handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
records = Entrez.read(handle)
retrieved = set(records["IdList"])

# Compute precision@5
precision = len(gold & retrieved) / 5
print("Baseline retrieved:", retrieved)
print(f"Baseline precision@5 = {precision:.2f}")