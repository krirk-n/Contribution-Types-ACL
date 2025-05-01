import pandas as pd
from tqdm import tqdm
from azure_openai import AzureOpenAIClient
import argparse
import os
import tiktoken
import asyncio

# === Load Prompt Template ===
with open("scripts/prompt_template.txt", "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()

def count_tokens(messages, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    total = 0
    for m in messages:
        total += 4
        for k, v in m.items():
            total += len(encoding.encode(v))
    total += 2
    return total

def build_user_prompt(abstract, max_length=None):
    return PROMPT_TEMPLATE.format(abstract=abstract[:max_length])

def parse_response(content):
    try:
        label_line = content.strip()
        labels = [label.strip() for label in label_line.split(",") if label.strip() in list("ABCDEFG")]
        if len(labels) > 2:
            labels = labels[:2]
        return set(labels), content
    except Exception:
        return set(), content

async def label_papers(input_path, output_path, model="gpt-4o", sample_size=None, year=None):
    ext = os.path.splitext(input_path)[-1]
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

    # === Filter by year if specified ===
    if year:
        df = df[df["year"] == str(year)]
        print(f"Filtered to year {year}. {len(df)} rows remain.")

    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampling {sample_size} papers from filtered dataset...")

    azure_client = AzureOpenAIClient()
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling papers"):
        messages = [{"role": "user", "content": build_user_prompt(row['abstract'])}]
        try:
            input_tokens = count_tokens(messages, model)
            content = await azure_client.get_chat_completion(messages, temperature=0.0, model=model)
            output_tokens = len(tiktoken.encoding_for_model(model).encode(content))
            total_tokens = input_tokens + output_tokens

            label_set, raw = parse_response(content)

            results.append({
                "acl_id": row["acl_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "year": row["year"],
                "A": "A" in label_set,
                "B": "B" in label_set,
                "C": "C" in label_set,
                "D": "D" in label_set,
                "E": "E" in label_set,
                "F": "F" in label_set,
                "G": "G" in label_set,
                "raw_response": raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            })

        except Exception as e:
            print(f"[ERROR] Paper {row['acl_id']} at index {idx}: {e}")
            results.append({
                "acl_id": row["acl_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "year": row["year"],
                "A": False, "B": False, "C": False,
                "D": False, "E": False, "F": False, "G": False,
                "raw_response": str(e),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            })

    output_ext = os.path.splitext(output_path)[-1]
    if output_ext == ".csv":
        pd.DataFrame(results).to_csv(output_path, index=False)
    elif output_ext == ".parquet":
        pd.DataFrame(results).to_parquet(output_path, index=False)
    else:
        raise ValueError("Unsupported output file format. Use .csv or .parquet")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label paper contributions using Azure OpenAI")
    parser.add_argument("--input", default="data/acl-publication-info.64k.parquet", help="Input .csv or .parquet file")
    parser.add_argument("--output", help="Output file path (if not provided, a name will be auto-generated)")
    parser.add_argument("--model", default="gpt-4o", help="Model to use (e.g., gpt-4o, o3-mini)")
    parser.add_argument("--sample", type=int, required=True, help="Number of rows to sample (required)")
    parser.add_argument("--year", type=int, help="Optional publication year filter")
    args = parser.parse_args()

    # === Auto-generate output path if not provided ===
    if args.output is None:
        sample_part = f"sample{args.sample}"
        year_part = f"{args.year}" if args.year else "all-years"
        args.output = f"result/labelled_{args.model}_{year_part}_{sample_part}.csv"

    asyncio.run(label_papers(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        sample_size=args.sample,
        year=args.year
    ))

# example usage:
# python scripts/label_contributions.py --model gpt-4o --sample 10 --year 2012