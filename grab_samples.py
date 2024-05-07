import torch
import numpy as np
import pandas as pd
import os
import argparse
from transformers import AutoTokenizer
from pydev.env1.projects.miapipeline.models import MODEL_MAP, print_available_models

def pick_sample(DF, metric = 'cosine_similarity'):
    if metric not in DF.columns:
        raise ValueError(f"Metric '{metric}' is not present in the DataFrame columns.")    
    top_indices = DF[metric].nlargest(10).index.tolist()
    return top_indices

# Console output is too long
def write_to_file(filename, idx, prefix_text, suffix_text, output_text):
    try:
        with open(filename, 'a', encoding='utf-8') as file:
            file.write("=" * 50 + "\n")
            file.write(f"Index {idx}\n")
            file.write(f"||Detokenized prefix|| \n{prefix_text}\n")
            file.write(f"||Detokenized suffix|| \n{suffix_text}\n")
            file.write(f"||Detokenized output|| \n{output_text}\n\n")
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError occurred: {e}. Some characters may not be properly saved.")
        
def main():
    parser = argparse.ArgumentParser(description="Load and process a model using specified parameters.")
    parser.add_argument("--model", type=str, required=True, help="The name of the model to load (e.g., CobraMamba/mamba-gpt-3b-v3).")
    parser.add_argument("--metric", type=str, required=True, help="The metric to use (e.g., cosine_similarity).")
    parser.add_argument("--prefix_path", type=str, default="{args.model}_Prefix.pt", help="Path to the prefix map file.")
    parser.add_argument("--output_path", type=str, default="{args.model}_Output.pt", help="Path to the output file.")
    parser.add_argument("--metric_fn", type=str, default="{args.model}_Analysis.csv", help="The filename of the metrics CSV.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    metrics = pd.read_csv(os.path.join('Metrics', args.metric_fn))

    print(f"Loaded metrics: {metrics.head()}")
    print(f"Using model: {args.model}")
    print(f"Using metric: {args.metric}")

    # Picking the top 10 for metric args.metric
    idxs = pick_sample(metrics, args.metric)
    
    prefix_data = torch.load(os.path.join('Data', args.prefix_path))
    output_data = torch.load(os.path.join('Output', args.output_path))
    
    for idx in idxs:
        suffix_text = prefix_data[idx]["text"]        
        prefix = prefix_data[idx]["input_ids"]
        prefix_text = tokenizer.decode(prefix)

        output = output_data.at[idx, "Data"]
        
        try:
            output_text = tokenizer.decode(output)
            write_to_file(f"{args.model}_{args.metric}.txt", idx, prefix_text, suffix_text, output_text)
        except:
            write_to_file(f"{args.model}_{args.metric}.txt", idx, prefix_text, suffix_text, output_text=f"ERROR IN OUTPUT:{output}")


if __name__ == "__main__":
    main()

