import pandas as pd

df = pd.read_csv("data/courses_processed.csv")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())