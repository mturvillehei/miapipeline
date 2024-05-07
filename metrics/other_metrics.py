import numpy as np
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