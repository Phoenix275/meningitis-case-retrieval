"""Checks that the PubMed corpus and the saved metadata agree."""

GOLD_QUERY_1 = {"40622531", "40696623", "40705280", "40656496", "40622525"}


def test_every_record_has_title_abstract_and_pmid(records):
    assert records, "data.jsonl is empty"
    for rec in records:
        prompt = rec["prompt"]
        assert prompt.startswith("Title: ")
        assert "\nAbstract: " in prompt
        assert prompt.endswith("Retrieve similar cases:")
        assert rec["metadata"]["pmid"].isdigit()


def test_pmids_are_unique(records):
    pmids = [rec["metadata"]["pmid"] for rec in records]
    assert len(pmids) == len(set(pmids))


def test_metadata_matches_corpus_order(records, metadata):
    # query_cases.py maps FAISS row i to metadata[i], so order must line up.
    assert len(metadata) == len(records)
    for rec, meta in zip(records, metadata):
        assert meta["pmid"] == rec["metadata"]["pmid"]
        assert meta["prompt"] == rec["prompt"]


def test_first_gold_query_is_fully_covered(records):
    pmids = {rec["metadata"]["pmid"] for rec in records}
    assert GOLD_QUERY_1 <= pmids
