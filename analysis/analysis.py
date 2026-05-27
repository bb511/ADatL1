from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "data" / "metrics" / "train_metrics_by_gamma.csv"

df = pd.read_csv(CSV_PATH)

df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
df = df.dropna(subset=["gamma"])
df = df.sort_values("gamma")

print(df[["gamma"]])
print(df["gamma"].dtype)

metric = "ascore_operational"

plt.figure(figsize=(10, 6))
plt.plot(df["gamma"], df[metric], marker="o")
plt.xlabel("Gamma")
plt.ylabel(metric)
plt.title(f"{metric} vs Gamma")
plt.grid(True)
plt.xscale("log")
plt.tight_layout()
plt.show()

print("done")