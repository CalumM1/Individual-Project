from first_pass_retrieval import run, DATASETS

if __name__ == "__main__":
    # run("antique_test", **DATASETS["antique_test"])
    for dataset in DATASETS:
        run(dataset, **DATASETS[dataset])