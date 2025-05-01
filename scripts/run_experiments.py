import subprocess
from pathlib import Path

YEARS = list(range(2014, 2023)) # 2012 to 2022 inclusive
MODEL = "gpt-4o"
N_SAMPLES = 100
INPUT_PATH = "data/acl-publication-info.64k.parquet"
SCRIPT_PATH = "scripts/label_contributions.py"
OUTPUT_DIR = Path("result/yearly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for year in YEARS:
    output_file = OUTPUT_DIR / f"labelled_{MODEL}_{year}_sample{N_SAMPLES}.csv"
    cmd = [
        "python", SCRIPT_PATH,
        "--input", INPUT_PATH,
        "--model", MODEL,
        "--year", str(year),
        "--sample", str(N_SAMPLES),
        "--output", str(output_file)
    ]
    print(f"\n[Running] {year}: {output_file}")
    subprocess.run(cmd, check=True)

print("\n✅ All experiments completed.")

# example usage:
# python scripts/run_experiments.py