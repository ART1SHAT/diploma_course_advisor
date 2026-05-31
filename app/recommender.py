import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

class CourseRecommender:

    def __init__(self):
        
        BASE_DIR = Path(__file__).resolve().parent.parent

        csv_path = BASE_DIR / "data" / "courses_processed.csv"

        self.df = pd.read_csv(csv_path)

        print("Loading model...")

        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.df = pd.read_csv(
            "data/courses_processed.csv"
        )

        self.texts = (
            self.df["title"].fillna("") +
            " " +
            self.df["description"].fillna("")
        ).tolist()

        print("Building embeddings...")

        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True
        )

        print("Ready")

    def recommend(self, query, top_k=5):

        query_embedding = self.model.encode([query])

        scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        self.df["similarity"] = scores

        result = self.df.sort_values(
            by="similarity",
            ascending=False
        )

        return result.head(top_k)