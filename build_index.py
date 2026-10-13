from pathlib import Path
import pyterrier as pt 


def build_index(index_path: str, dataset: str):
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
        if not doc.get("text", "").strip():
            continue
        yield doc


def main():
    build_index("antique_train.terrier", "irds:antique/train")
    build_index("antique_test.terrier", "irds:antique/test")
    build_index("trec_fair2021_eval.terrier", "irds:trec-fair/2021/eval")
    build_index("trec_fair2021_train.terrier", "irds:trec-fair/2021/train")
    build_index("trec_fair2022_train.terrier", "irds:trec-fair/2022/train")


if __name__ == "__main__":
    main()