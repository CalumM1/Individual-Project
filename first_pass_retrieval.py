from pathlib import Path
 
import pyterrier as pt
from pyterrier.measures import nDCG, AP, RR
 
WMODELS = {"bm25": "BM25", "dirichlet": "DirichletLM", "tfidf": "TF_IDF"}
 

DATASETS = {
    "trec_fair2021_train": {
        "index": "trec-fair-2021.terrier",
        "dataset": "irds:trec-fair/2021/train",
        "variant": "text",
    },
    "trec_fair2022_train": {
        "index": "trec-fair-2022.terrier",
        "dataset": "irds:trec-fair/2022/train",
        "variant": "text",
    },
    "antique_test": {
        "index": "antique_train.terrier",
        "dataset": "irds:antique/test",
        "rel": 3,
    },
    "beir_fiqa_test": {
        "index": "beir_fiqa_train.terrier",
        "dataset": "irds:beir/fiqa/test",
    },
    "beir_quora_test": {
        "index": "beir_quora_test.terrier",
        "dataset": "irds:beir/quora/test",
    },
    "cord19_trec_covid": {
        "index": "cord19_trec_covid.terrier",
        "dataset": "irds:cord19/trec-covid",
        "variant": "title",
    },
    "msmarco_doc_dl2020": {
        "index": "msmarco-document_trec-dl-2020_judged.terrier",
        "dataset": "irds:msmarco-document/trec-dl-2020/judged",
    },
}
 
 
def run(name, index, dataset, rel=1, variant=None):
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
 
    ds = pt.get_dataset(dataset)
    topics, qrels = ds.get_topics(variant=variant), ds.get_qrels()
    topics["query"] = topics["query"].str.replace(r"[^\w\s]", " ", regex=True)
 
    idx = pt.IndexFactory.of(f"./indexes/{index}")
 
    pt.Experiment(
        {n: pt.terrier.Retriever(idx, wmodel=w, num_results=1000) for n, w in WMODELS.items()},
        topics, qrels,
        eval_metrics=[nDCG @ 10, AP(rel=rel), RR(rel=rel)],
        save_dir=str(out),
        save_mode="reuse",
    )
 
    topics.to_parquet(out / "topics.parquet")
    qrels.to_parquet(out / "qrels.parquet")




