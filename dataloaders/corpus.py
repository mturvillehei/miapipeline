from datasets import load_dataset, load_from_disk
import os
import random
    
def dl_corpus():
    
    Corpus_en = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
    print(f"Corpus shape is {Corpus_en['train'].shape}")
    Corpus_shuffled = Corpus_en['train'].shuffle(seed=random())
    return Corpus_shuffled

def load_Corpus(crop=False, save=True, size="all"):
    """
    Load a subset of the Wikipedia dataset with additional control options.

    Parameters:
        crop (bool): Whether to crop the dataset to a specific size.
        save (bool): Whether to save the processed dataset to disk.
        size (int): Number of rows to retain if cropping.
        
    Returns:
        Dataset: The shuffled dataset.
    """

    if os.path.exists("./Data/Corpus_{size}"):
        print(f"Corpus dataset Corpus_{size} found in ./Data")
        Corpus_shuffled = load_from_disk(os.path.join("Data", f"Corpus_{size}"))
        
    else:
        Corpus_shuffled = dl_corpus()
        if save:
            Corpus_shuffled.save_to_disk("./Data/Corpus_{size}")
    # Crop the dataset if requested
    
    if crop and size is not "all":
        Corpus_shuffled = Corpus_shuffled.select(range(size))

    return Corpus_shuffled
