import argparse
from datasets import DATALOADERS, ACCESSDATA, available_datasets
from Models.models import TOKENIZERS
import torch

def load_dataset(dataset, size):
    try:
        return DATALOADERS[dataset](size)
    except KeyError:
        available_datasets(dataset)
        exit()
        

def gen_prefix_map(dataset, tokenizer, ):
    ''' 
    
    Parameters:
        dataset = The dataset that's being used.
        tokenizer = Tokenizer for the model being used. If we're using an API model, we don't tokenize the prefix, suffix or original. -- Note: we can tokenize after the fact if the tokenizer is not none.
        prefix_tokens = The number of tokens in the prefix. Default is 10.
        prefix_loc = Where to begin
        size = Number of samples to generate prefixes for.
        
    Here, I'm splitting up the return into prefix_map and original_map. This allows us to load the prefix_map 
    individually, which should be less computationally dense. We'd like to preserve the original text, but
    it doesn't need to be in the same .pt file.
    
    Returns:
        prefix_map = {
            "idx": Index, # Using index as a key to help with discontinuous generation (such as we saw in the project)
            {
                "prefix_ids" = list
            }
        }
        Original Data
        original_map = {
            "idx": Index, # Using index as a key to help with discontinuous generation (such as we saw in the project)
            {
                "label" =  str # label of the sample, e.g. for Corpus this is the article title
                "info" = str # Other aspects of the sample, e.g. for Corpus this is the url
                "sample_ids" = list,
                "suffix_ids" = list # Tokens of the original text with the prefix removed. 
            }
        }
    '''    
    
    return

def main():
    # Parsing configurations
    parser = argparse.ArgumentParser(description="Generate a prefix map for a given model's tokenizer, using a given dataset.")
    parser.add_argument("--model", type=str, required=True, help="The model to use for tokenizing (e.g., mamba-3b).")
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset to load (e.g., Corpus.")
    parser.add_argument("--prefix_length", type=int, default=10, help="Length of the generated prefixes.")
    parser.add_argument("--prefix_location", type=str, default="start", help="Where the prefix is chosen from. Options are: 'start', 'random', or an integer (e.g. '9'). If the start token is longer than sample_length - prefix_length, the last possible token will be chosen (i.e. sample_length - prefix_length).")
    parser.add_argument("--size", type=str, default="all", help="Number of samples to include in the prefix map.")
    
    args = parser.parse_args()

    # Load the dataset here
    full_dataset = load_dataset(args.dataset, args.size)
    
    # Grabbing the columns we want as a tuple (right now, 'title' and 'text')
    dataset = ACCESSDATA[args.dataset](full_dataset)
    
    # Prefix map gen here
    tokenizer = TOKENIZERS(args.model)
    prefix_map, original_map = gen_prefix_map(dataset, tokenizer, args.prefix_length, args.prefix_location, args.size)
        
    # Save here. We should create a new directory containing the prefix_map and original_map for each generated prefix map.
    # ...

if __name__ == "__main__":
    main()

