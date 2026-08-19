from pathlib import Path

from gateway.evals import enumeration_recall, write_sample_pdas
from gateway.packs.rag import HybridIndex, ingest_workspace


def test_demurrage_negatives_are_not_substring(tmp_path: Path):
    write_sample_pdas(tmp_path, n=12)
    hits = []
    for p in tmp_path.glob("PDA-*.txt"):
        if "demurrage" in p.read_text(encoding="utf-8").lower():
            hits.append(p.name)
    assert len(hits) == 4


def test_hybrid_enumeration(tmp_path: Path):
    write_sample_pdas(tmp_path, n=12)
    warning = ingest_workspace(tmp_path)
    assert warning["files_scanned"] == 12
    assert warning["no_text_layer_count"] == 0
    idx = HybridIndex(tmp_path)
    result = idx.search("demurrage", limit=40)
    rec = enumeration_recall(idx, "demurrage", result["paths"])
    assert rec["ground_truth"] == 4
    assert rec["recall"] >= 0.90
    assert rec["precision"] >= 0.90
    assert rec["cause"] is None


def test_full_sentence_query_still_enumerates(tmp_path: Path):
    write_sample_pdas(tmp_path, n=12)
    ingest_workspace(tmp_path)
    idx = HybridIndex(tmp_path)
    result = idx.search("List every PDA mentioning demurrage", limit=12)
    rec = enumeration_recall(idx, "demurrage", result["paths"])
    assert rec["recall"] >= 0.90
    assert rec["precision"] >= 0.90
    assert rec["returned"] == rec["ground_truth"]


def test_incomplete_enumeration_fails(tmp_path: Path):
    write_sample_pdas(tmp_path, n=12)
    ingest_workspace(tmp_path)
    idx = HybridIndex(tmp_path)
    rec = enumeration_recall(idx, "demurrage", ["PDA-001-NEPTUNE.txt"])
    assert rec["cause"] == "incomplete_enumeration"
    assert rec["recall"] < 0.90


def test_search_from_other_thread(tmp_path: Path):
    write_sample_pdas(tmp_path, n=12)
    ingest_workspace(tmp_path)
    idx = HybridIndex(tmp_path)
    box: dict = {}

    def worker():
        box["r"] = idx.search("demurrage", limit=40)

    import threading

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)
    assert box["r"]["paths"]
    rec = enumeration_recall(idx, "demurrage", box["r"]["paths"])
    assert rec["recall"] >= 0.90
