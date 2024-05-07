from Levenshtein import distance


def levenshtein_distance(A, B):
    # Requires String form
    # https://en.wikipedia.org/wiki/Levenshtein_distance
    return distance(' '.join(B), ' '.join(A))

# Implement modified levenshtein

def hamming_distance(A, B):
    if len(A) != len(B):
        longer, shorter = (A, B) if len(A) > len(B) else (B, A)
        min_hamming = 0  
        for i in range(len(longer) - len(shorter) + 1):
            snippet = longer[i:i + len(shorter)]
            current_hamming = sum(el1 != el2 for el1, el2 in zip(snippet, shorter))
            min_hamming = min(min_hamming, current_hamming)
        return min_hamming
    else:
        return sum(el1 != el2 for el1, el2 in zip(A, B))

