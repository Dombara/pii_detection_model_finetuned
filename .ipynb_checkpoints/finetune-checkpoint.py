import pandas as pd
from datasets import Dataset

# Load the TSV file
df = pd.read_csv("indian_pii_dataset_corrected_labeled.tsv", sep="\t")

# Group by Sentence-ID to create lists of words and labels
grouped = df.groupby("Sentence-ID").agg({
    "Word": lambda x: list(x),
    "Label": lambda x: list(x)
}).reset_index()

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(grouped[["Word", "Label"]])