from Models.models import MODELS, MODEL_TYPES
from typing import Callable
from torch.nn.utils.rnn import pad_sequence

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