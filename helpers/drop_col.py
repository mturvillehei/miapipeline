import pandas as pd
import os


if __name__ == "__main__":

    fn = "Gemma_Data_Analysis.csv"
    col = ["Prefix_IDs", "Suffix_IDs", "Output_IDs"]
    header = True

    DF = pd.read_csv(os.path.join('temp', fn))

    DF.drop(columns=col).to_csv(os.path.join('temp', fn))