from dataloaders.corpus import load_Corpus


if __name__ == "__main__":
    corpus = load_Corpus(crop=True, save=True, size=500000)