from dataloaders.corpus import load_Corpus



DATALOADERS= {
    'Corpus': lambda size: load_Corpus(size), # Need to check this
    'Common_Crawl': lambda size: print(f"Not implemented")
}

# Modular return to allow for different dataset types.
ACCESSDATA = {
    'Corpus': lambda dataset: (dataset['text'], dataset['title'], dataset['url']),
    'Common_Crawl': lambda dataset: ("Not implemented", "Not Implemented", "Not Implemented")
}

def available_datasets(dataset=None):
    if dataset is None:
        print("Available models (arguments):")
    else:
        print(f"Invalid dataset passed as arg: {dataset}.")
        print("Available models (arguments):")
        for ds_key in DATALOADERS:
            print(f"- {ds_key}")
