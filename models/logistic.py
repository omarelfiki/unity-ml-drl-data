import pandas as pd 

df = pd.read_csv("../data/combined_results.csv")

df["steps_to_threshold"] = pd.to_numeric(df["steps_to_threshold"], errors="coerce")
df["time_to_threshold"] = pd.to_numeric(df["time_to_threshold"], errors="coerce")

df["reached_threshold"] = (
    (df["steps_to_threshold"] > 0) &
    (df["time_to_threshold"] > 0)
).astype(int)

print(df["reached_threshold"].value_counts())
