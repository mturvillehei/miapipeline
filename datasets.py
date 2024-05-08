from dataloaders.corpus import load_Corpus

DATALOADERS= {
    'Corpus': lambda size: load_Corpus(size),
    'Common_Crawl': lambda size: print(f"Not implemented")
}

def available_datasets(dataset=None):
    if dataset is None:
        print("Available models (arguments):")
    else:
        print(f"Invalid dataset passed as arg: {dataset}.")
        print("Available models (arguments):")
        for ds_key in DATALOADERS:
            print(f"- {ds_key}")
