from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, LlamaForCausalLM, LlamaTokenizer
from Models.claude_api import API_CALL
import torch
MODEL_MAP = {
    'gemma-2b': "google/gemma-2b",
    'open_llama_3b': "openlm-research/open_llama_3b",
    'mamba-3b': "CobraMamba/mamba-gpt-3b-v3",
    'claude_api': 'API_CALL'
}

# Since we're using lambda functions, we can modify this to accept arbitrary args depending on the model type. 
MODELS = {
    'gemma-2b': lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_MAP['gemma-2b'], device_map="auto") if torch.cuda.is_available()
        else AutoModelForCausalLM.from_pretrained(MODEL_MAP['gemma-2b']),

    'open_llama_3b': lambda: LlamaForCausalLM.from_pretrained(
        MODEL_MAP['open_llama_3b'], torch_dtype=torch.float16, device_map='auto', offload_folder="offload"),

    'mamba-3b': lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_MAP['mamba-3b'], trust_remote_code=True, torch_dtype=torch.float16),
    
    'claude-API': lambda sysprompt, payload, *args: API_CALL(sysprompt, payload, *args)
}
TOKENIZERS= {
    'gemma-2b': lambda: AutoTokenizer.from_pretrained(MODEL_MAP['gemma-2b']),
    'open_llama_3b': lambda: LlamaTokenizer.from_pretrained(MODEL_MAP['open_llama_3b']),
    'mamba-3b': lambda: AutoTokenizer.from_pretrained(MODEL_MAP['mamba-3b']),
    'claude-API': None
}

def available_models():
    print("Available models (arguments):")
    for model_name in MODEL_MAP:
        print(f"- {model_name}")

