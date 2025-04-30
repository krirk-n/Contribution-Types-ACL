1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Download the full dataset:
    [ACL-OCL Dataset](https://huggingface.co/datasets/WINGNUS/ACL-OCL/resolve/main/acl-publication-info.74k.v2.parquet)

3. Preprocess the dataset:
    Run the preprocessing script to generate `acl-publication-info.64k.parquet`, which contains rows with non-null `abstract` and `full_text` fields:
    ```bash
    python scripts/preprocess.py
    ```