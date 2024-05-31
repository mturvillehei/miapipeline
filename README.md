# miapipeline
For easy testing of string membership and memorization in LLMs.

# Pipeline structure.

## Prefix Generation

First, a prefix map is required as input to the model. Generating a prefix map is described below.

1. prefix_generation.py uses the DATALOADER for the dataset args.dataset
    1. Runs the DATALOADER for the dataset chosen (args.dataset) 
    2. If the dataset is not on disk, the dataset is downloaded by the dataloader
    3. The dataloader then returns the dataset with sample count = size. If size is not specified, the entire dataset is returned. Results are shuffled automatically.
2. The prefix_generation script then generates a tokenized prefix map for length "prefix_tokens". 
    1. The prefix start location by default is the start token index 0. 
    2. Other possible choices are "random" and integer positions. If the integer position exceeds the final position - prefix length, the last possible location is chosen.
3. Prefix map is then stored in "".\Prefixes 

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

Map location name format
The prefix map and original map are stored in a new directory within the "Prefixes" directory for each execution of prefix_generation.py. The directory name is formatted as follows:
`{model}_{dataset}_{prefix_tokens}_{prefix_loc}_{timestamp}`

model: The model used for tokenizing (e.g., mamba-3b).
dataset: The name of the dataset used (e.g., Corpus).
prefix_tokens: The token length of the generated prefixes.
prefix_loc: Where the prefix is chosen from ('start', 'random', or an integer position).
timestamp: The timestamp of when the script was run, in the format YYYYMMDD_HHMMSS.

For example, a directory name might look like:
`mamba-3b_Corpus_10_start_20230510_143015`
Within this directory, the prefix map is stored as `prefix_map.pt` and the original map is stored as `original_map.pt`.

## Model Prompting

The `model_prompting.py` script is responsible for generating strings using a given model and prefix map. It takes the following arguments:

- `--prefix_map`: The name of the prefix map to use (e.g., 'prefix_map_dir'). This should correspond to a folder in the `Prefixes` directory.
- `--model`: The model to use for generating strings (e.g., 'mamba-3b').
- `--num_strings`: The total number of strings to generate. If the number of strings is larger than the dataset size, prefixes will be used repeatedly.
- `--batch_size`: The number of strings to generate in each batch (default is 10).
- `--max_length`: The maximum output length from the model (default is 20). The actual output length may not reach this length.

The script performs the following steps:

1. It parses the command-line arguments using `argparse`.
2. It loads the prefix map and original map from the specified `prefix_map` directory.
3. It initializes the specified model based on the `model` argument.
4. It processes the dataset in batches using the `batch_prompt` function:
   - If `num_strings` is less than or equal to the dataset size, it generates outputs for the requested number of strings.
   - If `num_strings` is greater than the dataset size, it implements wraparound to continue generating outputs from the beginning of the dataset until the desired number of strings is reached.
   - For each batch, it calls the appropriate `process_batch` function based on the model type (API or local).
   - The `process_batch` function passes the batch to the corresponding `process_api_batch` or `process_local_batch` function.
   - The `process_api_batch` function sends each entry's text to the API model and returns the generated output.
   - The `process_local_batch` function pads the input tokens, passes them to the local model, and returns the generated output.
5. The generated output for each batch is collected and returned as the final output.

The script supports two types of models:

- API models: These models require an API key and are executed on a server. The generated output is obtained by sending the input text to the API.
- Local models: These models run locally on your machine. The input tokens are padded and passed to the local model for generation.

The `load_model` function and `MODEL_TYPES` dictionaries are used to map the model names to their corresponding functions and types.

To run the script, use the following command:

```bash
$ python prompt.py --prefix_map prefix_map_dir --model mamba-3b --num_strings 100 --max_length 50
```

This will generate 100 strings using the 'mamba-3b' model and the prefix map stored in the 'prefix_map_dir' directory. The maximum output length from the model will be set to 50 tokens.

If the number of requested strings (`num_strings`) is greater than the dataset size, the script will implement wraparound and continue generating outputs from the beginning of the dataset until the desired number of strings is reached.

The generated output will be returned as a list of dictionaries, where each dictionary contains the generated tokens for a single input.

