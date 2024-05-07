import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate a prefix map.")
    parser.add_argument("--model", type=str, required=True, help="The name of the dataset to load (e.g., Corpus500k.")
    parser.add_argument("--dataset", type=str, required=True, help="The model to use for tokenizing (e.g., mamba-3b).")
    # ...
if __name__ == "__main__":
    main()

