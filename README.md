# Meningitis Case-Similarity Retrieval & Summary Agent

An NLP retrieval pipeline that matches a free-text patient description against published meningitis case reports and returns the most similar cases with one-sentence treatment and outcome summaries.

**Published and presented at the MIT Undergraduate Research Technology Conference (URTC).**

## What it does

A clinician types a short patient profile, for example:

> 18 y/o with fever, neck stiffness, CSF neutrophils >1000

The system embeds that description, searches a FAISS index built over PubMed case reports, returns the top-5 nearest cases with their PMIDs, and generates a one-sentence summary of how each case was treated and how it turned out.

The goal is decision support at the point of reading: rather than keyword-searching PubMed and skimming abstracts, a clinician sees semantically similar prior cases immediately.

## Results

| Metric | Score |
| --- | --- |
| Retrieval precision@5 | 0.24 |
| ROUGE-1 (summaries) | 0.39 |
| ROUGE-L (summaries) | 0.26 |

Retrieval is evaluated against five hand-labeled gold-standard queries with known relevant PMIDs (`evaluate_retrieval.py`). `baseline.py` runs the same evaluation using a plain PubMed keyword query, so the embedding approach can be compared against ordinary search rather than reported in isolation.

## How it works

1. **Corpus construction** (`build_jsonl.py`) — queries PubMed via Biopython/Entrez for meningitis case reports and reviews, extracts titles and abstracts, and writes `data.jsonl`.
2. **Indexing** (`build_index.py`) — embeds each record with OpenAI `text-embedding-ada-002` and builds a FAISS `IndexFlatL2` over the vectors.
3. **Retrieval** (`query_cases.py`) — embeds the query profile and returns the k nearest case reports by L2 distance.
4. **Summarization** (`app.py`) — for each retrieved case, generates a one-sentence treatment/outcome summary with GPT-3.5-turbo at temperature 0.
5. **Evaluation** (`evaluate_retrieval.py`, `evaluate_summaries.py`, `baseline.py`) — precision@5 against gold PMIDs, ROUGE against reference summaries, and a keyword-search baseline.

## Stack

Python, OpenAI embeddings and chat completions, FAISS, Biopython (Entrez), Streamlit, NumPy.

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="your-key-here"
export NCBI_API_KEY="your-ncbi-key"   # optional, raises PubMed rate limits
export NCBI_EMAIL="you@example.com"   # required by NCBI

python build_jsonl.py     # fetch case reports from PubMed
python build_index.py     # build the FAISS index
streamlit run app.py      # launch the demo
```

Command-line retrieval without the UI:

```bash
python3 query_cases.py "3 month old infant with fever, irritability, bulging fontanelle"
```

Evaluation:

```bash
python3 evaluate_retrieval.py   # precision@5 vs gold PMIDs
python3 baseline.py             # keyword-search baseline for comparison
python3 evaluate_summaries.py   # ROUGE scores
```

## Limitations

Precision@5 of 0.24 reflects a small corpus (50 PubMed records) and a strict gold standard where only a handful of PMIDs count as correct for each query. The index uses exact L2 search, which is fine at this scale but would need an approximate index to grow. Summaries are generated from abstracts rather than full text, so they inherit whatever the abstract omits. This is a research prototype, not a clinical tool, and nothing it outputs should inform patient care.

## Author

Tegh Bindra — [github.com/Phoenix275](https://github.com/Phoenix275)
