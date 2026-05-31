import requests
import pandas as pd

BASE_URL = "https://stepik.org/api/courses"


def get_courses(page=1):
    response = requests.get(
        BASE_URL,
        params={
            "page": page
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["courses"]


def collect_courses(pages=5):
    result = []

    for page in range(1, pages + 1):
        print(f"Page {page}")

        courses = get_courses(page)

        for course in courses:
            result.append({
                "id": course.get("id"),
                "title": course.get("title"),
                "description": course.get("summary"),
                "price": course.get("price"),
                "language": course.get("language"),
                "url": f"https://stepik.org/course/{course.get('id')}"
            })

    return result


if __name__ == "__main__":
    courses = collect_courses(300)

    df = pd.DataFrame(courses)

    df.to_csv(
        "data/courses.csv",
        index=False
    )

    print(df.head())
    print(f"Saved {len(df)} courses")