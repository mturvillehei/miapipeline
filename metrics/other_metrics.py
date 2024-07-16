from typing import Any, Tuple, Dict, List
import numpy as np
import math

class KullbackLeibler:
    def __init__(self):
        self.dataset_freq = {}
        self.dataset_length = 0

    def update_dataset(self, data):
        for char in data:
            self.dataset_freq[char] = self.dataset_freq.get(char, 0) + 1
            self.dataset_length += 1

    def calculate_divergence(self, string):
        # Count the frequency of each character in the string
        string_freq = {}
        for char in string:
            string_freq[char] = string_freq.get(char, 0) + 1

        # Normalize the frequencies to get the probability distribution
        string_prob = {}
        string_length = len(string)
        for char, freq in string_freq.items():
            string_prob[char] = freq / string_length

        # Normalize the frequencies of the dataset to get the probability distribution
        dataset_prob = {}
        for char, freq in self.dataset_freq.items():
            dataset_prob[char] = freq / self.dataset_length

        # Calculate the Kullback-Leibler divergence
        kl_divergence = 0
        for char in string_prob:
            if char in dataset_prob:
                kl_divergence += string_prob[char] * math.log(string_prob[char] / dataset_prob[char])

        return kl_divergence
    
# ToDo: add function to each model in Models.models for vocab_size 
class GT_Estimator:
    def __init__(self, vocab_size, n_gram_length):
        self.vocab_size = vocab_size
        self.n_gram_length = n_gram_length  # Add this line
        self.total_count = 0
        self.count_dict = {}
        self.count_of_counts = {}
        self.smoothed_count_of_counts = {}
        self.discount = {}
        self.smoothed_probs = {}

    def update(self, word: str) -> None:
        """
        Update the counts for a given word.
        
        Args:
            word (str): The word to update counts for.
        """
        self.total_count += 1
        if word in self.count_dict:
            old_count = self.count_dict[word]
            self.count_dict[word] += 1
            new_count = self.count_dict[word]
            
            self.count_of_counts[old_count] = self.count_of_counts.get(old_count, 0) - 1
            self.count_of_counts[new_count] = self.count_of_counts.get(new_count, 0) + 1
            
            if self.count_of_counts[old_count] == 0:
                del self.count_of_counts[old_count]
        else:
            self.count_dict[word] = 1
            self.count_of_counts[1] = self.count_of_counts.get(1, 0) + 1

    def smooth_count_of_counts(self) -> None:
        """
        Smooth the count-of-counts using linear regression in log space.
        """
        counts = sorted(self.count_of_counts.keys())
        log_counts = [math.log(c) for c in counts]
        log_count_of_counts = [math.log(self.count_of_counts[c]) for c in counts]
        
        n = len(counts)
        sum_x = sum(log_counts)
        sum_y = sum(log_count_of_counts)
        sum_xy = sum(x * y for x, y in zip(log_counts, log_count_of_counts))
        sum_xx = sum(x * x for x in log_counts)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        for c in range(1, max(counts) + 1):
            self.smoothed_count_of_counts[c] = math.exp(intercept + slope * math.log(c))

    def calculate_discount(self) -> None:
        """
        Calculate the discount values for each count.
        """
        for c in range(1, max(self.count_of_counts.keys()) + 1):
            n_c = self.count_of_counts.get(c, 0)
            n_c1 = self.count_of_counts.get(c + 1, 0)
            
            if c == 0:
                self.discount[c] = 0
            elif n_c1 == 0:
                self.discount[c] = 1 - (c + 1) * self.smoothed_count_of_counts.get(c + 1, 0) / (c * self.smoothed_count_of_counts[c])
            else:
                self.discount[c] = (c + 1) * n_c1 / (c * n_c) - 1

    def calculate_smoothed_probs(self) -> None:
        """
        Calculate the smoothed probabilities for each word in the vocabulary.
        """
        total_discount = sum(self.discount[c] * c * self.count_of_counts.get(c, 0) for c in self.count_of_counts)
        p0 = total_discount / (self.total_count * self.vocab_size)
        
        for word, count in self.count_dict.items():
            if count == 0:
                self.smoothed_probs[word] = p0
            else:
                self.smoothed_probs[word] = max(0, count - self.discount.get(count, 0)) / self.total_count

    def estimate(self) -> None:
        """
        Perform the Good-Turing estimation.
        """
        self.smooth_count_of_counts()
        self.calculate_discount()
        self.calculate_smoothed_probs()

    def get_probability(self, word: str) -> float:
        """
        Get the smoothed probability for a given word.
        
        Args:
            word (str): The word to get the probability for.
        
        Returns:
            float: The smoothed probability of the word.
        """
        return self.smoothed_probs.get(word, self.smoothed_probs.get('UNK', 0))
    
    def estimate_unseen_ngrams(self):
        """
        Estimate the number of unseen n-grams based on the Good-Turing estimator.
        
        Returns:
            float: The estimated number of unseen n-grams.
        """
        N1 = self.count_of_counts.get(1, 0)  # Number of n-grams that appear exactly once
        N = sum(count * freq for count, freq in self.count_of_counts.items())  # Total number of observed n-grams
        
        if N == 0:
            return 0 
        
        p0 = N1 / N

        total_possible_ngrams = self.vocab_size ** self.n_gram_length
        
        unseen_ngrams = total_possible_ngrams * p0
        
        return unseen_ngrams
    
    def __call__(self, words: List[str]) -> Tuple[Dict[str, float], float]:
        """
        Update the estimator with new words and return their probabilities and unseen n-grams estimate.
        
        Args:
            words (List[str]): A list of words to update and get probabilities for.
        
        Returns:
            Tuple[Dict[str, float], float]: A tuple containing:
                - A dictionary of words and their smoothed probabilities
                - The estimated number of unseen n-grams
        """
        for word in words:
            self.update(word)
        self.estimate()
        
        probabilities = {word: self.get_probability(word) for word in words}
        unseen_ngrams = self.estimate_unseen_ngrams()
        
        return probabilities, unseen_ngrams
    
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

# Perplexity is in other_metrics.py because the model/tokenizer is required as an arg
def perplexity():
    return