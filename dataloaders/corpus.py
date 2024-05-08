from datasets import load_dataset, load_from_disk
import os
import random
    
def dl_corpus():
    Corpus_en = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
    Corpus_shuffled = Corpus_en['train'].shuffle(seed=random.randint(0, 10000))
    return Corpus_shuffled
def load_Corpus(size="all", save=True, crop=False):
    """
    Load a subset of the Wikipedia dataset with additional control options.

    Parameters:
    size (int): Number of rows to retain if cropping.
    save (bool): Whether to save the processed dataset to disk.
    crop (bool): Whether to crop the dataset to a specific size.

    Returns:
    Dataset: The shuffled dataset.
    """
    if crop and size != "all":
        dataset_name = f"Corpus_{size}"
    else:
        dataset_name = "Corpus_all"
        size = "all"

    if os.path.exists(f"./Data/{dataset_name}"):
        print(f"Corpus dataset {dataset_name} found in ./Data")
        Corpus_shuffled = load_from_disk(os.path.join("Data", dataset_name))
    else:
        Corpus_shuffled = dl_corpus()
        if crop and size != "all":
            Corpus_shuffled = Corpus_shuffled.select(range(size))
        if save:
            Corpus_shuffled.save_to_disk(f"./Data/{dataset_name}")

    return Corpus_shuffled
