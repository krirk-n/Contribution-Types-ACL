import pandas as pd
import os

# Define file paths
input_file = os.path.join('data', 'acl-publication-info.74k.v2.parquet')
output_file = os.path.join('data', 'acl-publication-64k.parquet')

# Check if input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file does not exist: {input_file}")

df = pd.read_parquet(input_file)

# Filter rows with non-null 'full_text' and 'abstract'
df_clean = df[df['full_text'].notna() & df['abstract'].notna()].reset_index(drop=True)

# Save the cleaned dataset
df_clean.to_parquet(output_file, index=False)