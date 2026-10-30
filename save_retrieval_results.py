from first_pass_retrieval import run, DATASETS

if __name__ == "__main__":

    for dataset in DATASETS:
        run(dataset, **DATASETS[dataset])