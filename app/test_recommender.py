from recommender import CourseRecommender

rec = CourseRecommender()

result = rec.recommend(
    "хочу стать дата аналитиком"
)

print(
    result[
        [
            "title",
            "category",
            "similarity"
        ]
    ]
)