from importlib import import_module
import pkgutil
import sys
from pathlib import Path

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


models_parent_dir = Path(__file__).resolve().parent
sys.path.append(str(models_parent_dir))

def load_model(model_name, **kwargs):
    try:
        model_module = import_module(f"models.{model_name.replace('-', '_')}")
        return model_module.load_model(**kwargs)
    except ImportError:
        raise ValueError(f"Invalid model name: {model_name}")

def load_tokenizer(model_name, **kwargs):
    try:
        model_module = import_module(f"models.{model_name.replace('-', '_')}")
        return model_module.load_tokenizer(**kwargs)
    except ImportError:
        raise ValueError(f"Invalid model name: {model_name}")

def available_models():
    print("Available models (arguments):")
    for model_name in MODEL_MAP:
        try:
            model_module = import_module(f".{model_name.replace('-', '_')}", package="models")
            print(f"- {model_name}")
        except ImportError as e:
            print(f"Error importing model: {model_name}. Error: {str(e)}")