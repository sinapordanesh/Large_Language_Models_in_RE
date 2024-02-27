import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("code_search_net", "python")

# Convert the training, validation, and test splits to Pandas DataFrames
train_df = pd.DataFrame(dataset['train'])
validation_df = pd.DataFrame(dataset['validation'])
test_df = pd.DataFrame(dataset['test'])

# Optionally, you can concatenate these DataFrames if you want a single CSV
full_df = pd.concat([train_df, validation_df, test_df])

# Save the DataFrame to a CSV file
full_df.to_csv("code_search_net_python.csv", index=False)
