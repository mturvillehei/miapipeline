import argparse
from datasets import DATALOADERS, available_datasets
def load_dataset(dataset, size):
    try:
        return DATALOADERS[dataset](size)
    except KeyError:
        available_datasets(dataset)
        exit()

def main():
    parser = argparse.ArgumentParser(description="Generate a prefix map for a given model's tokenizer, using a given dataset.")
    parser.add_argument("--model", type=str, required=True, help="The model to use for tokenizing (e.g., mamba-3b).")
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset to load (e.g., Corpus.")
    parser.add_argument("--prefix_tokens", type=str, required=True, help="Token length of the generated prefixes.")
    parser.add_argument("--prefix_loc", type=str, default=str(0), help="Where the prefix is chosen from. Options are: 'start', 'random', or an integer (e.g. '9'). If the start token is longer than sample_length - prefix_tokens, the last possible token will be chosen (i.e. sample_length - prefix_tokens).")
    parser.add_argument("--size", type=str, default="all", help="Number of samples to include in the prefix map.")
    
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.size)
    
    # ...
if __name__ == "__main__":
    main()

