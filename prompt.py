import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

def batch_tokenize(text_list, title_list, url_list, batch_size=1000):
    """ Tokenize the text in batches.

    Args:
        text_list (list): List of text strings.
        title_list (list): List of title strings.
        url_list (list): List of url strings.
        batch_size (int, optional): Batch size for tokenization. Defaults to 1000.

    Returns:
        tuple: Tuple of tokenized text, title, and url lists.
    """
    tokenizer = TreebankWordTokenizer()
    tokenized_text = []
    tokenized_title = []
    tokenized_url = []

    for i in range(0, len(text_list), batch_size):
        batch_text = text_list[i:i + batch_size]
        batch_title = title_list[i:i + batch_size]
        batch_url = url_list[i:i + batch_size]

        tokenized_batch_text = [tokenizer.tokenize(text) for text in batch_text]
        tokenized_batch_title = [tokenizer.tokenize(title) for title in batch_title]
        tokenized_batch_url = [tokenizer.tokenize(url) for url in batch_url]

        tokenized_text.extend(tokenized_batch_text)
        tokenized_title.extend(tokenized_batch_title)
        tokenized_url.extend(tokenized_batch_url)

    return tokenized_text, tokenized_title, tokenized_url

# Example usage:
text_list = ["Example text 1", "Example text 2", "Example text 3"]
title_list = ["Example title 1", "Example title 2", "Example title 3"]
url_list = ["http://example.com/1", "http://example.com/2", "http://example.com/3"]

dataset = (text_list, title_list, url_list)
tokenized_dataset = batch_tokenize(*dataset)
