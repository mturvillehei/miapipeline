from datasets import load_dataset
import os

def load_Corpus(crop=False, save=True, size="all"):
    """
    Load a subset of the Wikipedia dataset with additional control options.

    Parameters:
        crop (bool): Whether to crop the dataset to a specific size.
        save (bool): Whether to save the processed dataset to disk.
        size (int): Number of rows to retain if cropping.
    Returns:
        Dataset: The processed dataset.
    """

    if os.path.exists("./Data/Corpus_{size}"):
        print(f"Corpus dataset already found in ./Data")
        return
    
    Corpus_en = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
    print(f"Corpus shape is {Corpus_en['train'].shape}")

    Corpus_shuffled = Corpus_en['train'].shuffle(seed=1010)

    # Crop the dataset if requested
    if crop and size is not "all":
        Corpus_shuffled = Corpus_shuffled.select(range(size))

    # Save the dataset to disk if required
    if save:
        Corpus_shuffled.save_to_disk("./Data/Corpus_{size}")

    return Corpus_shuffled
