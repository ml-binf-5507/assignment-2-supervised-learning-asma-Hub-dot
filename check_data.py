import pandas as pd

# Load dataset
df = pd.read_csv("heart_disease_uci(1).csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== UNIQUE VALUES IN TARGET COLUMN (if exists) =====")
if "num" in df.columns:
    print(df["num"].unique())
    