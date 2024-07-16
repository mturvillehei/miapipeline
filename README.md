# miapipeline
For easy testing of string membership and memorization in LLMs.

# Pipeline structure.

## Prefix Generation

First, a prefix map is required as input to the model. Generating a prefix map is described below.

1. prefix_generation.py uses the DATALOADER for the specified dataset (args.dataset)
2. The script generates a tokenized prefix map for the given prefix length
3. Prefix map is stored in ".\Prefixes"

#### prefix_generation.py args:
**args**
* `--model`: The model to use for tokenizing (e.g., mamba-3b).
* `--dataset`: The name of the dataset to load (e.g., Corpus).
* `--prefix_tokens`: Token length of the generated prefixes.
* `--prefix_loc`: Where the prefix is chosen from. Options are: 'tart', 'random', or an integer (e.g., '9'). If the start token is longer than sample_length - prefix_tokens, the last possible token will be chosen (i.e., sample_length - prefix_tokens).
* `--size`: Number of samples to include in the prefix map. Default is "all".
* `--save`: Whether to save the dataset. Defaults to True.
* `--crop`: If the dataset is saved, whether the dataset should be limited to size. If this is False, then size is used in prefix map generation, but the whole dataset is saved (rather than just a subset).

```bash 
$ python prefix_generation.py --model mamba-3b --dataset corpus  
```

### Model options
There are two primary types of supported models - local and API. Local models will attempt to run on your PC. API models require an API key and are executed on server, and *will cost money*.

**Local models**
* `gemma-2b`: 
* `open_llama-3b`: 
* `mamba-3b`: 

**API models**:
* `claude-API`:

### Map location name format
Prefix maps are stored in the "Prefixes" directory with the following naming convention:
```
{model}_{dataset}_{prefix_tokens}_{prefix_loc}_{timestamp}
```
Example: mamba-3b_Corpus_10_start_20230510_143015

## Model Prompting

The `model_prompting.py` script is responsible for generating strings using a given model and prefix map. It takes the following arguments:

- `--prefix_map`: The name of the prefix map to use (e.g., 'prefix_map_dir'). This should correspond to a folder in the `Prefixes` directory.
- `--model`: The model to use for generating strings (e.g., 'mamba-3b').
- `--num_strings`: The total number of strings to generate. If the number of strings is larger than the dataset size, prefixes will be used repeatedly.
- `--batch_size`: The number of strings to generate in each batch (default is 10).
- `--max_length`: The maximum output length from the model (default is 20). The actual output length may not reach this length.

The script performs the following steps:
The script loads the prefix map, initializes the model, and generates strings in batches. It supports both API and local models.

```bash
$ python prompt.py --prefix_map <prefix_map directory> --model mamba-3b --num_strings 100 --max_length 50
```

This will generate 100 strings using the 'mamba-3b' model and the prefix map stored in the 'prefix_map_dir' directory. The maximum output length from the model will be set to 50 tokens.

If the number of requested strings (`num_strings`) is greater than the dataset size, the script will implement wraparound and continue generating outputs from the beginning of the dataset until the desired number of strings is reached.

The generated output will be returned as a list of dictionaries, where each dictionary contains the generated tokens for a single input.

## References

 - Scalable Extraction of Training Data from (Production) Language Models https://arxiv.org/abs/2311.17035
 - William A. Gale & Geoffrey Sampson (1995) Good‐turing frequency estimation without tears , Journal of Quantitative Linguistics, 2:3, 217-237, DOI: 10.1080/09296179508590051
