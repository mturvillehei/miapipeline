from Models.models import MODELS, MODEL_TYPES
from typing import Callable
from torch.nn.utils.rnn import pad_sequence

def process_batch(batch, model_fn, max_length, model_type):
    if model_type == "api":
        # For API models
        output = [model_fn(entry['text'], max_length) for entry in batch]
    else:
        # For other models
        input_ids = [entry['tokens'] for entry in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        if isinstance(model_fn, Callable):
            model = model_fn()
            output = model.generate(input_ids=input_ids, max_length=max_length)
        else:
            raise ValueError("Invalid model function")
    return output

def batch_prompt(dataset, model, batch_size, max_length):
    '''
    Parameters:
    dataset: Each row is a dict containing the prefix text and the prefix tokens: 'text' 'tokens'
    model: key of the model to use
    batch_size: batch size for model prompting
    max_length: max number of tokens to generate
    
    Returns:
    output: Each row is a dict containing the output tokens of the model.
    '''
    model_fn = MODELS[model]
    model_type = MODEL_TYPES[model]  # Get the model type
    output = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batch_output = process_batch(batch, model_fn, max_length, model_type)
        output.extend(batch_output)
    
    return output