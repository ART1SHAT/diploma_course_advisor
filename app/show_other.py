import pandas as pd

df = pd.read_csv("data/courses_processed.csv")

other = df[df["category"] == "Other"]

for i, row in other.head(100).iterrows():
    print(row["title"])