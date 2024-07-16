from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(**kwargs):
    return AutoModelForCausalLM.from_pretrained("CobraMamba/mamba-gpt-3b-v3", trust_remote_code=True, torch_dtype=torch.float16, **kwargs)

def load_tokenizer(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained("CobraMamba/mamba-gpt-3b-v3")

    def encode_func(input_text, max_length=100, **encode_kwargs):
        return tokenizer.encode(input_text, return_tensors="pt")[0]

    def decode_func(input_ids, **decode_kwargs):
        return tokenizer.decode(input_ids, skip_special_tokens=decode_kwargs.get('special_tokens', True))

    return tokenizer, encode_func, decode_func