from query_cases import find_similar_pmids

# Define your five gold‐standard queries and their true PMIDs
gold = {
    "18 year old with fever, neck stiffness, and CSF neutrophils >1000": {"40622531","40696623","40705280","40656496","40622525"},
    "16 year old girl with tender forehead swelling and fever, ultrasound suggesting Pott’s puffy tumor": {"40734865"},
    "60 year old with headaches and orbital pain, MRI showing frontal extra-axial mass invading bone": {"40735163"},
    "3 month old infant with high fever, irritability, bulging fontanelle, CSF culture positive for E coli": {"40622525"},
    "Patient with insidious onset ataxia and GFAP-IgG positive in CSF suggestive of autoimmune GFAP astrocytosis": {"40735311"}
}

def precision_at_5(predicted, truth):
    return len(predicted & truth) / 5.0

scores = []
for profile, true_set in gold.items():
    preds, _ = find_similar_pmids(profile, k=5)
    p5 = precision_at_5(set(preds), true_set)
    print(f"Query: {profile}\n  Predicted: {preds}\n  Precision@5 = {p5:.2f}\n")
    scores.append(p5)

avg = sum(scores) / len(scores)
print(f"Average precision@5 over {len(scores)} queries = {avg:.2f}")