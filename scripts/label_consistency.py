import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

MODEL_FILES = {
    "gpt4o": "result/labelled_gpt-4o_sample10.csv",
    "o3mini": "result/labelled_o3-mini_sample10.csv",
    "gpt4omini": "result/labelled_gpt-4o-mini_sample10.csv"
}

LABELS = list("ABCDEFG")

# === Load and merge all model outputs ===
dfs = []
for name, path in MODEL_FILES.items():
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)[["acl_id"] + LABELS].copy()
    elif path.endswith(".csv"):
        df = pd.read_csv(path)[["acl_id"] + LABELS].copy()
    else:
        raise ValueError(f"Unsupported file format for {path}")
    df.columns = ["acl_id"] + [f"{l}_{name}" for l in LABELS]
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="acl_id")

# === Per-paper Jaccard similarities between all model pairs ===
def get_labels(row, model):
    return {l for l in LABELS if row[f"{l}_{model}"]}

def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1.0

for m1, m2 in combinations(MODEL_FILES.keys(), 2):
    merged[f"jaccard_{m1}_{m2}"] = merged.apply(
        lambda row: jaccard(get_labels(row, m1), get_labels(row, m2)),
        axis=1
    )

# === Plot histogram of Jaccard scores ===
plt.figure()
for m1, m2 in combinations(MODEL_FILES.keys(), 2):
    plt.hist(merged[f"jaccard_{m1}_{m2}"], bins=10, range=(0, 1), alpha=0.6, label=f"{m1} vs {m2}")

plt.title("Jaccard Similarity Between Models")
plt.xlabel("Jaccard Similarity")
plt.ylabel("Number of Papers")
plt.legend()
plt.tight_layout()
plt.savefig("figures/jaccard_similarity_histogram.png")
plt.close()

# === Per-label agreement (percentage) ===
agreement = {}
for label in LABELS:
    votes = merged[[f"{label}_{m}" for m in MODEL_FILES.keys()]]
    agreement[label] = (votes.nunique(axis=1) == 1).mean()

# Plot label-wise agreement bar chart
plt.figure()
sns.barplot(x=list(agreement.keys()), y=list(agreement.values()))
plt.title("Label-wise Full Agreement Across Models")
plt.ylabel("Proportion of Papers with Full Agreement")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/label_agreement_bar_chart.png")
plt.close()

# === Save merged for further inspection ===
merged.to_parquet("merged_model_outputs.parquet")
print("Saved merged file with Jaccard scores to: merged_model_outputs.parquet")
print("Saved plots to: jaccard_similarity_histogram.png and label_agreement_bar_chart.png")
