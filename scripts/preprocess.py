import pandas as pd
import os

def filter_empty_abstract_and_full_text():
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

    return df_clean

def filter_only_main_conference(df):
    # Filter rows where 'acl_id' includes specific patterns or starts with 'P'
    patterns = ['emnlp-main', 'acl-main', 'naacl-main']
    df_filtered = df[df['acl_id'].str.contains('|'.join(patterns)) | df['acl_id'].str.startswith('P')].reset_index(drop=True)
    
    return df_filtered

def filter_only_specific_field(df, field):
    # Filter rows where 'title' or 'abstract' contains the specified field (case-insensitive)
    df_filtered = df[df['title'].str.contains(field, case=False, na=False) | 
                     df['abstract'].str.contains(field, case=False, na=False)].reset_index(drop=True)
    
    return df_filtered

def sample_yearly(df, seed=42, num_samples=100, start_year=2013, end_year=2022):
    # Sample rows for each year
    sampled_dfs = []
    for year in range(start_year, end_year + 1):
        df_year = df[df['year'] == str(year)]
        if not df_year.empty:
            sampled_df = df_year.sample(n=min(num_samples, len(df_year)), random_state=seed)
            sampled_dfs.append(sampled_df)

    # Concatenate all sampled dataframes
    if sampled_dfs:
        df_sampled = pd.concat(sampled_dfs).reset_index(drop=True)
    else:
        df_sampled = pd.DataFrame()  # Return an empty DataFrame if no data is sampled

    return df_sampled

if __name__ == "__main__":
    # Filter empty abstracts and full texts
    df = filter_empty_abstract_and_full_text()
    # df = pd.read_parquet('data/acl-publication-info.64k.parquet')

    # Filter only main conference papers
    df_filtered = filter_only_main_conference(df)
    print(df_filtered[:10])  # Display the first 10 rows of the filtered dataframe
    df_filtered.to_parquet('data/acl-publication-info.main.parquet', index=False)

    # Sample yearly data
    df_sampled = sample_yearly(df_filtered, num_samples=100)
    print(f"Number of rows after sampling: {len(df_sampled)}")
    df_sampled.to_parquet('data/acl-publication-info.main.yearlysampled100.parquet', index=False)

    # df_filtered = pd.read_parquet('data/acl-publication-info.main.parquet')
    # print(f"Number of rows: {len(df_filtered)}")

    # Filter only specific field (e.g., "NLP")
    df_filtered = filter_only_specific_field(df, "machine translation")
    # print(df_filtered[:10])  # Display the first 10 rows of the filtered dataframe
    print(f"Number of rows: {len(df_filtered)}")
    df_filtered.to_parquet('data/acl-publication-info.machine-translation.parquet', index=False)

    # df_filtered = pd.read_parquet('data/acl-publication-info.machine-translation.parquet')
    # print(f"Number of rows: {len(df_filtered)}")

    # Sample yearly data
    df_sampled = sample_yearly(df_filtered, num_samples=100)
    print(f"Number of rows after sampling: {len(df_sampled)}")
    df_sampled.to_parquet('data/acl-publication-info.machine-translation.yearlysampled100.parquet', index=False)
