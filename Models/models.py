from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, LlamaForCausalLM, LlamaTokenizer
from Models.claude_api import OPUS_API_CALL
import torch

MODEL_MAP = {
    'gemma-2b': "google/gemma-2b",
    'open-llama-3b': "openlm-research/open_llama_3b",
    'mamba-3b': "CobraMamba/mamba-gpt-3b-v3",
    'claude-opus-api': 'API_CALL'
}

MODEL_TYPES = {
    'gemma-2b': "local",
    'open-llama-3b': "local",
    'mamba-3b': "local",
    'claude-opus-api': "api"
}

def load_model(model_name, **kwargs):
    if model_name == 'gemma-2b':
        return AutoModelForCausalLM.from_pretrained(MODEL_MAP[model_name], device_map="auto", **kwargs) if torch.cuda.is_available() else AutoModelForCausalLM.from_pretrained(MODEL_MAP[model_name], **kwargs)
    elif model_name == 'open-llama-3b':
        return LlamaForCausalLM.from_pretrained(MODEL_MAP[model_name], torch_dtype=torch.float16, device_map='auto', offload_folder="offload", **kwargs)
    elif model_name == 'mamba-3b':
        return AutoModelForCausalLM.from_pretrained(MODEL_MAP[model_name], trust_remote_code=True, torch_dtype=torch.float16, **kwargs)
    elif model_name == 'claude-opus-api':
        return OPUS_API_CALL
    else:
        raise ValueError(f"Invalid model name: {model_name}")

# When filling this out, prefix_generation expects input_ids
def load_tokenizer(model_name, **kwargs):
    if model_name == 'claude-opus-api':
        return None, None, None
    
    # Implemented
    elif model_name == 'mamba-3b':
        tokenizer = AutoTokenizer.from_pretrained("CobraMamba/mamba-gpt-3b-v3")
        
        def encode_func(input_text, max_length=100, **encode_kwargs):
            return tokenizer.encode(input_text, return_tensors="pt")[0]
        
        def decode_func(input_ids, **decode_kwargs):
            return tokenizer.decode(input_ids, skip_special_tokens=decode_kwargs.get('special_tokens', True))
        
        return tokenizer, encode_func, decode_func
    
    # This needs to be fixed
    elif model_name == 'gemma-2b':
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name], **kwargs)
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
    
    # This needs to be implemented still
    elif model_name == 'open-llama-3b':
        return
    else:
        print(f"Tokenizer for model {model_name} is not implemented.")
        exit()

def available_models():
    print("Available models (arguments):")
    for model_name in MODEL_MAP:
        print(f"- {model_name}")