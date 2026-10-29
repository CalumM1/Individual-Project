from pathlib import Path
import pyterrier as pt
import sys


def build_index(dataset, index_path):
    dataset = pt.get_dataset(dataset)

    path = Path(__file__).resolve().parent / "indexes" / index_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        indexer = pt.IterDictIndexer(str(path))
        indexer.index(dedup_iter(dataset.get_corpus_iter()))


    return str(path)


def dedup_iter(corpus_iter):
    seen = set()
    for doc in corpus_iter:
        if doc["docno"] in seen:
            continue
        seen.add(doc["docno"])

        if not (doc.get("text") or "").strip():
            parts = [doc.get(f) or "" for f in ("title", "abstract", "body")]
            doc = {**doc, "text": " ".join(p for p in parts if p).strip()}

        yield doc


def main():
    if len(sys.argv) != 3:
        print("build_index.py expects the dataset and the name of the index directory")
    build_index(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
