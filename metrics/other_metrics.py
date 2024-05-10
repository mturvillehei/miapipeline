from typing import Any
import numpy as np

class GT_Estimator():
    """
    A class for implementing the Good-Turing frequency estimation algorithm.

    Args:
        vocab_size (int): The size of the vocabulary, i.e., the total number of unique words or tokens in the dataset.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        total_count (int): The total count of all words or tokens encountered so far.
        count_dict (dict): A dictionary storing the count of each word or token in the vocabulary.
            - Keys: words/tokens
            - Values: corresponding counts
        count_of_counts (dict): A dictionary keeping track of the number of words/tokens that have a specific count.
            - Keys: counts
            - Values: number of words/tokens with that count
        smoothed_count_of_counts (dict): A dictionary storing the smoothed count of counts using the Good-Turing estimation.
            - Keys: counts
            - Values: smoothed count of counts
        discount (dict): A dictionary storing the discount values for each count based on the Good-Turing estimation.
            - Keys: counts
            - Values: discount values
        smoothed_probs (dict): A dictionary storing the smoothed probabilities for each word/token in the vocabulary.
            - Keys: words/tokens
            - Values: smoothed probabilities
    """
    # We will probably need to initialize with an arbitrarily small, or arbitrarily large, vocab_size. 
    # Implementing in the same style as Carlini et al. (2023) will need to account for very long vocab sequences (50-grams, etc.).
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.total_count = 0
        self.count_dict = {}
        self.count_of_counts = {}
        self.smoothed_count_of_counts = {}
        self.discount = {}
        self.smoothed_probs = {}
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

def modified_hamming_distance(A, B):
    if len(A) != len(B):

        longer, shorter = (A, B) if len(A) > len(B) else (B, A)
        hammings = []

        for i in range(len(longer) - len(shorter) + 1):
            snippet = longer[i:i + len(shorter)]
            current_hamming = sum(el1 != el2 for el1, el2 in zip(snippet, shorter))
            # Normalizing by length L of the shorter string
            # Generally shorter is the output, so this is what we want
            # If the suffix is shorter than the output, this will still
            # prioritize longer matches.
            current_hamming /= len(shorter)
            hammings.append(current_hamming)
        return np.min(hammings)
    
    else:
        return [sum(el1 != el2 for el1, el2 in zip(A, B))] / len(A)


def kullback_leibler(A, B): 
    # Implement over the entire dataset, then compute kullback-liebler of the A and B to see if they match?
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    return

