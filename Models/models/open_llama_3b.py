from transformers import LlamaForCausalLM, LlamaTokenizer

def load_model(**kwargs):
    return LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b", torch_dtype=torch.float16, device_map='auto', offload_folder="offload", **kwargs)

def load_tokenizer(**kwargs):
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b", **kwargs)

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