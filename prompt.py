from Models.models import MODELS, MODEL_TYPES
from typing import Callable
from torch.nn.utils.rnn import pad_sequence
import argparse

def process_api_batch(batch, model_fn, max_length):
    output = [model_fn(entry['text'], max_length) for entry in batch]
    return output

def process_local_batch(batch, model_fn, max_length):
    input_ids = [entry['tokens'] for entry in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    if isinstance(model_fn, Callable):
        model = model_fn()
        output = model.generate(input_ids=input_ids, max_length=max_length)
    else:
        raise ValueError("Invalid model function")
    return output

BATCH_PROCESSORS = {
    "api": process_api_batch,
    "local": process_local_batch
}

def process_batch(batch, model_fn, max_length, model_type):
    process_fn = BATCH_PROCESSORS.get(model_type)
    if process_fn is None:
        raise ValueError(f"Invalid model type: {model_type}")
    return process_fn(batch, model_fn, max_length)


def batch_prompt(dataset, model, batch_size, num_strings, max_length):
    '''
    Parameters:
    dataset: Each row is a dict containing the prefix text and the prefix tokens: 'text' 'tokens'
    model: key of the model to use
    batch_size: batch size for model prompting
    num_strings: total number of strings to use
    max_length: max number of tokens to generate
    
    Returns:
    output: Each row is a dict containing the output tokens of the model.
    '''

    model_fn = MODELS[model]
    model_type = MODEL_TYPES[model]  # Get the model type
    output = []
    

    if num_strings <= len(dataset):

        for i in range(0, num_strings, batch_size):
            batch = dataset[i:i+batch_size]
            batch_output = process_batch(batch, model_fn, max_length, model_type)
            output.extend(batch_output)
    # For if num_strings is more than the dataset length
    # Frankly, this is fragile and shouldn't be used lol
    else:
        num_iterations = (num_strings + len(dataset) - 1) // len(dataset)
        for _ in range(num_iterations):
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                batch_output = process_batch(batch, model_fn, max_length, model_type)
                output.extend(batch_output)
                if len(output) >= num_strings:
                    break
            if len(output) >= num_strings:
                break
        output = output[:num_strings]


    return output

if __name__ == "__main__":

    # Parsing configurations
    parser = argparse.ArgumentParser(description="Generate strings using a given model and prefix map.")
    parser.add_argument("--prefix_map", type=str, required=True, help="The name of the prefix map to use (e.g., 'my_prefix_map').")
    parser.add_argument("--model", type=str, required=True, help="The model to use for generating strings (e.g., 'mamba-3b').")
    parser.add_argument("--num_strings", type=int, required=True, help="The total number of strings to generate. If N is larger than the dataset size, prefixes will be used repeatedly.") # Need to implement wraparound here.
    parser.add_argument("--batch_size", type=int, default=10, help="The number of strings to generate in each batch.")
    parser.add_argument("--max_length", type=int, default = 20, help="Max output length from the model. Output length may not reach this length.")

    args = parser.parse_args()

    # Unload the variables
    prefix_map = args.prefix_map
    model = args.model
    num_strings = args.num_strings
    batch_size = args.batch_size
    max_length = args.max_length

    output = batch_prompt(prefix_map, model, batch_size, num_strings, max_length)