from dataloaders.corpus import load_Corpus

DATALOADERS= {
    'corpus': lambda size, save, crop: load_Corpus(size, save, crop), 
    'common_Crawl': lambda size, save, crop: print(f"Not implemented")
}

# Expected datatype is a Dict with the keys "text", "label", "info" (optional)
ACCESSDATA = {
    'corpus': lambda dataset: {'text' : dataset['text'], 'label': dataset['title'], 'info': dataset['url']},
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
