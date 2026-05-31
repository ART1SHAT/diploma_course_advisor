import pandas as pd

df = pd.read_csv("data/courses_processed.csv")

print(df["category"].value_counts())