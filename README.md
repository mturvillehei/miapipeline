# miapipeline
For easy testing of string membership in LLMs.

# Pipeline structure.

### Run prefix_generation.py with args listed below, which performs the following:


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


### Run Main.py 


### Model options
There are two primary types of supported models - local and API. Local models will attempt to run on your PC. API models require an API key and are executed on server, and *will cost money*.

**Local models**
* `gemma-2b`: 
* `open_llama-3b`: 
* `mamba-3b`: 

**API models**:
* `claude-API`:  
