from dataloaders.corpus import load_Corpus



DATALOADERS= {
    'corpus': lambda size, save, crop: load_Corpus(size, save, crop), 
    'common_Crawl': lambda size, save, crop: print(f"Not implemented")
}

# Modular return to allow for different dataset types.
ACCESSDATA = {
    'corpus': lambda dataset: (dataset['text'], dataset['title'], dataset['url']),
    'common_crawl': lambda dataset: ("Not implemented", "Not Implemented", "Not Implemented")
}

def available_datasets(dataset=None):
    if dataset is None:
        print("Available models (arguments):")
    else:
        print(f"Invalid dataset passed as arg: {dataset}.")
        print("Available models (arguments):")
        for ds_key in DATALOADERS:
            print(f"- {ds_key}")
