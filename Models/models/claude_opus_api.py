from anthropic_api import OPUS_API_CALL

def load_model(**kwargs):
    return OPUS_API_CALL

def load_tokenizer(**kwargs):
    return None, None, None