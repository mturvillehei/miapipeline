from Models.models import MODEL_MAP, MODELS, TOKENIZERS
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a prefix map for a given model's tokenizer, using a given dataset.")
    parser.add_argument("--model", type=str, required=True, help="The model to use for tokenizing (e.g., mamba-3b).")
    parser.add_argument("--prefix_map", type=str, required=True, help="The name of the prefix map. Ignore the file extension (e.g. 'Corpus_500000'). Prefix maps are saved in ./Prefixes/")
    parser.add_argument("--prefix_tokens", type=str, required=True, help="Token length of the generated prefixes.")
    parser.add_argument("--prefix_loc", type=str, default=str(0), help="Where the prefix is chosen from. Options are: 'start', 'random', or an integer (e.g. '9'). If the start token is longer than sample_length - prefix_tokens, the last possible token will be chosen (i.e. sample_length - prefix_tokens).")
    parser.add_argument("--size", type=str, default="all", help="Number of samples to include in the prefix map.")
    
    args = parser.parse_args()


# Would be interesting if we ran this with a visualizer for data as it's being generated
# i.e. after the sample is generated we can look at the cosine similarity in the cli dump
