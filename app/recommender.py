import numpy as np
import pandas as pd

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class CourseRecommender:

    def __init__(self):

        print("Loading model...")

        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        base_dir = Path(__file__).resolve().parent.parent

        csv_path = base_dir / "data" / "courses_processed.csv"

        self.df = pd.read_csv(csv_path)

        self.df = self.df.fillna("")

        self.texts = (
            self.df["title"].astype(str)
            + " "
            + self.df["description"].astype(str)
        ).tolist()

        embeddings_path = (
            base_dir
            / "models"
            / "course_embeddings.npy"
        )

        if embeddings_path.exists():

            print("Loading embeddings...")

            self.embeddings = np.load(
                embeddings_path
            )

        else:

            print("Building embeddings...")

            self.embeddings = self.model.encode(
                self.texts,
                show_progress_bar=True
            )

            np.save(
                embeddings_path,
                self.embeddings
            )

            print("Embeddings saved")

        print("Ready")

    def recommend(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query]
        )

        similarity_scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        result_df = self.df.copy()

        result_df["similarity"] = similarity_scores

        # float колонка
        result_df["bonus"] = 0.0

        query_lower = query.lower()

        boosts = {

            "Data Science": [
                "аналитик",
                "анализ",
                "данные",
                "data"
            ],

            "Artificial Intelligence": [
                "ai",
                "нейросеть",
                "gpt",
                "llm"
            ],

            "Python": [
                "python"
            ],

            "Web Development": [
                "frontend",
                "backend",
                "web",
                "javascript"
            ],

            "DevOps": [
                "docker",
                "kubernetes",
                "devops"
            ]
        }

        for category, keywords in boosts.items():

            if any(
                keyword in query_lower
                for keyword in keywords
            ):

                mask = (
                    result_df["category"]
                    == category
                )

                result_df.loc[
                    mask,
                    "bonus"
                ] += 0.15

        result_df["final_score"] = (
            result_df["similarity"]
            + result_df["bonus"]
        )

        result_df = result_df.sort_values(
            by="final_score",
            ascending=False
        )

        return result_df.head(top_k)