from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(**kwargs):
    return AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", **kwargs) if torch.cuda.is_available() else AutoModelForCausalLM.from_pretrained("google/gemma-2b", **kwargs)

def load_tokenizer(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", **kwargs)
    tokenizer.pad_token = tokenizer.eos_token

    def encode_func(input_text, max_length=10, **encode_kwargs):
        return tokenizer.encode_plus(
            text=input_text,
            add_special_tokens=encode_kwargs.get('add_special_tokens', True),
            max_length=max_length,
            padding=encode_kwargs.get('padding', "max_length"),
            truncation=encode_kwargs.get('truncation', True),
            return_tensors=encode_kwargs.get('return_tensors', "pt")
        )['input_ids']

    def decode_func(input_ids, **decode_kwargs):
        return tokenizer.decode(input_ids)

    return tokenizer, encode_func, decode_func