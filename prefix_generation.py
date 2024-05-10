import argparse
import random
import torch
import datetime
import os
from implemented_datasets import DATALOADERS, ACCESSDATA, available_datasets
from Models.models import TOKENIZERS

def load_dataset(dataset, size, save, crop):
    try:
        return DATALOADERS[dataset](size, save, crop)
    except KeyError:
        available_datasets(dataset)
        exit()

def gen_prefix_map(dataset, tokenizer, prefix_length, prefix_location, size):
    
    ''' 

    Parameters:
        dataset: Dict, keys are "text", "label", "info" (optional)
        tokenizer (Tokenizer or None): Tokenizer for the model being used. If we're using an API model, the prefix, suffix, and original are not tokenized. 
        prefix_length (int): The number of tokens in the prefix.
        prefix_location (str): Location to start prefix extraction. See documentation for available options.
        size (int): Number of samples to generate prefixes for.
        
    Returns:

        prefix_map (list): Each row is a dict containing the prefix text and the prefix tokens: 'text' 'tokens'
        
        original_map (list): Each row is a dict containing the original text, label (e.g. article title), info: 'text', 'label', 'info',

    ''' 

    combined_data = list(zip(dataset['text'], dataset['label'], dataset.get('info', [None] * len(dataset['text']))))
    random.shuffle(combined_data)
    if size != "all":
        try:
            size = int(size)
            if size < 0:
                raise ValueError("Size must be a positive integer or 'all'.")
        except ValueError as e:
            raise ValueError(f"Invalid size value: {e}")

        size = min(size, len(combined_data))
        combined_data = combined_data[:size]

    prefix_map = []
    original_map = []

    tokenizer.pad_token = tokenizer.eos_token
    
    for idx, (text, label, info) in enumerate(combined_data):
        print(f"Text: {text}")
        print(f"label: {label}")
        print(f"info: {info}")
        print("---")
        text_ids = tokenizer.tokenize(text)
        
        if prefix_location == "start":
            start_index = 0
        elif prefix_location == "random":
            start_index = random.randint(0, max(0, len(text_ids) - prefix_length))
        else:  
            start_index = min(prefix_location, len(text_ids) - prefix_length)
        
        prefix_ids = text_ids[start_index:start_index + prefix_length]
        prefix_text = tokenizer.decode

        prefix_map.append({'text': prefix_text, 'tokens': prefix_ids})
        original_map.append({'text': text, 'label': label, 'info': info }) # Could also save the tokenized original here. 

    return prefix_map, original_map

def main():
    # Parsing configurations
    parser = argparse.ArgumentParser(description="Generate a prefix map for a given model's tokenizer, using a given dataset.")
    parser.add_argument("--model", type=str, required=True, help="The model to use for tokenizing (e.g., mamba-3b).")
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset to load (e.g., Corpus.")
    parser.add_argument("--prefix_length", type=int, default=10, help="Length of the generated prefixes.")
    parser.add_argument("--prefix_location", type=str, default="start", help="Where the prefix is chosen from. Options are: 'start', 'random', or an integer (e.g. '9').")
    parser.add_argument("--size", type=str, default="all", help="Number of samples to include in the prefix map.")
    parser.add_argument("--save", type=bool, default=True, help="Whether to save the dataset. Defaults to True.")
    parser.add_argument("--crop", type=bool, default=False, help="If the dataset is saved, whether the dataset should be limited to size. If this is False, then size is used in prefix map generation, but the whole dataset is saved (rather than just a subset). ")
    
    args = parser.parse_args()

    # Load the dataset here
    full_dataset = load_dataset(args.dataset, args.size, args.save, args.crop)
    
    # Grabbing the columns we want as a tuple (right now, 'title' and 'text')
    dataset = ACCESSDATA[args.dataset](full_dataset)
    
    # Prefix map gen here
    tokenizer, = TOKENIZERS[args.model]
    prefix_map, original_map = gen_prefix_map(dataset, tokenizer, args.prefix_length, args.prefix_location, args.size)
        
   # Save the maps to their own directory.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"Prefixes/{args.model}_{args.dataset}_{args.prefix_length}_{args.prefix_location}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    prefix_file = f"{output_dir}/prefix_map.pt"
    original_file = f"{output_dir}/original_map.pt"

    torch.save(prefix_map, prefix_file)
    torch.save(original_map, original_file)

    print(f"Saved files to directory: {output_dir}")
    print(f"Prefix map file: {prefix_file}")
    print(f"Original map file: {original_file}")

if __name__ == "__main__":
    main()

