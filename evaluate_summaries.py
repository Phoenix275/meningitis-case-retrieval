import os
import json
import time
from openai import OpenAI
from rouge_score import rouge_scorer

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Reference summaries for evaluation
target_profiles = {
    "40622531": "Patients receive supportive care (hydration, analgesics, monitoring), and most make a full neurological recovery, though severe cases may develop encephalitis with lasting sequelae.",
    "40696623": "Both elderly patients underwent surgical drainage plus culture-guided IV antibiotics and achieved complete neurological recovery without residual deficits.",
    "40705280": "Incorporating NGS and multiplex PCR panels shortened time-to-diagnosis by 24–48 hours, enabling earlier targeted therapies and a 15% reduction in ICU length-of-stay.",
    "40622525": "Neonatal E. coli meningitis was treated with high-dose cefotaxime, leading to complete recovery without sequelae.",
    "40735311": "High-dose corticosteroid treatment produced dramatic symptom resolution within 2–4 weeks in all three patients, avoiding misdiagnosis pitfalls and long-term deficits."
}

# Load prompts from data.jsonl into a dict
records = {}
with open("data.jsonl", "r") as f:
    for line in f:
        rec = json.loads(line)
        pmid = rec["metadata"]["pmid"]
        records[pmid] = rec["prompt"]

# Prepare ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

# Evaluate summaries
results = []
for pmid, ref_sum in target_profiles.items():
    raw_prompt = records.get(pmid)
    if not raw_prompt:
        print(f"Skipping PMID {pmid}: not found in data.jsonl")
        continue
    # Remove retrieval instruction, keep only Title+Abstract
    prompt_text = raw_prompt.split("\n\nRetrieve similar cases:")[0]

    # Few-shot prompt with two examples
    few_shot = """
Example 1:
Title: Pterygomaxillary space infection complicated by meningitis due to Streptococcus constellatus.
Abstract: Streptococcus constellatus, part of the normal oral flora, can cause deep-space head and neck infections. We describe two elderly male patients who developed pterygomaxillary space abscesses that extended intracranially, resulting in bacterial meningitis. Both underwent surgical drainage of the abscess and received a course of intravenous antibiotics tailored to culture sensitivities. They achieved complete neurological recovery with no residual deficits.
Summary: Both patients underwent surgical drainage and culture-guided IV antibiotic therapy, resulting in full neurological recovery without deficits.

Example 2:
Title: E. coli Meningitis.
Abstract: E. coli meningitis is a rare but serious infection in neonates. This case report details presentation with fever and bulging fontanelle, CSF culture confirmation, and treatment with high-dose cefotaxime. The infant recovered fully without neurological sequelae.
Summary: Neonatal E. coli meningitis was treated with high-dose cefotaxime, leading to complete recovery without sequelae.

Now summarize this case report in one sentence (treatment + outcome):
{full_prompt}
""".strip()

    # Insert actual case prompt
    content = few_shot.replace("{full_prompt}", prompt_text)

    # Call ChatGPT to generate summary
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=60
    )
    auto_sum = resp.choices[0].message.content.strip()
    time.sleep(1)

    # Compute ROUGE scores
    scores = scorer.score(ref_sum, auto_sum)
    results.append({
        "pmid": pmid,
        "reference": ref_sum,
        "auto": auto_sum,
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    })

# Print results as a Markdown table
print("| PMID | Reference Summary | Auto-Summary | ROUGE-1 | ROUGE-L |")
print("|------|-------------------|--------------|---------|---------|")
for r in results:
    print(f"| {r['pmid']} | {r['reference']} | {r['auto']} | {r['rouge1']:.2f} | {r['rougeL']:.2f} |")