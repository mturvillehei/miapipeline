# This is a temp script to manually load datasets. In general this should be handled via prefix_generation.py on execution.

from dataloaders.corpus import load_Corpus

if __name__ == "__main__":
    corpus = load_Corpus(crop=True, save=True, size=500000)