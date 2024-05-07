from collections import Counter
import torch

def k_cluster(A, B):

    return 1 - (len(set(B) & set(A))/len(set(B) | set(A)))


def jaccard_similarity(A, B): 
    sset = set(B)
    oset = set(A)
    numerator = len(sset & oset)
    denominator = len(sset | oset)
    return numerator / denominator


def dice_coefficient(A, B):
    """
    Calculate the Dice coefficient between the prefix and B.
    """
    A_counter = Counter(A)
    B_counter = Counter(B)
    intersection = sum((A_counter & B_counter).values())

    union = sum(A_counter.values()) + sum(B_counter.values()) - intersection
    if union == 0:
        return 0  # Return 0 if union is zero to avoid division by zero
    else:
        return 2 * intersection / union

def cosine_similarity(A, B):
    counter1 = Counter(A)
    counter2 = Counter(B)
    
    all_tokens = set(counter1.keys()).union(set(counter2.keys()))
    
    vector1 = torch.tensor([counter1[token] for token in all_tokens])
    vector2 = torch.tensor([counter2[token] for token in all_tokens])
    
    cosine_sim = torch.nn.functional.cosine_similarity(vector1.float(), vector2.float(), dim=0)
    
    return cosine_sim.item()