import pandas as pd

df = pd.read_csv("data/courses.csv")

df = df.fillna("")

df["full_text"] = (
    df["title"].astype(str) + " " +
    df["description"].astype(str)
)

categories = {

    "Python": [
        "python",
        "django",
        "flask",
        "fastapi",
        "pandas"
    ],

    "Data Science": [
        "data science",
        "data scientist",
        "аналитик данных",
        "анализ данных",
        "data analyst",
        "machine learning",
        "ml",
        "numpy",
        "pandas",
        "sklearn",
        "scikit",
        "matplotlib",
        "seaborn"
    ],

   "Artificial Intelligence": [
        "ai",
        "artificial intelligence",
        "нейросеть",
        "нейронная сеть",
        "llm",
        "gpt",
        "chatgpt",
        "transformer"
    ],

    "Deep Learning": [
        "deep learning",
        "tensorflow",
        "keras",
        "pytorch",
        "cnn",
        "rnn"
    ],

    "Computer Vision": [
        "computer vision",
        "opencv",
        "image processing"
    ],

    "NLP": [
        "nlp",
        "natural language",
        "обработка текста",
        "bert"
    ],

    "Web Development": [
        "html",
        "css",
        "javascript",
        "frontend",
        "backend",
        "web development"
    ],

    "React": [
        "react",
        "redux",
        "next.js"
    ],

    "Vue": [
        "vue",
        "vuejs"
    ],

    "Angular": [
        "angular",
        "typescript"
    ],

    "Java": [
        "java",
        "spring",
        "hibernate"
    ],

    "C#": [
        "c#",
        ".net",
        "asp.net"
    ],

    "C++": [
        "c++",
        "qt"
    ],

    "Go": [
        "golang",
        "go language"
    ],

    "Mobile Development": [
        "android",
        "ios",
        "flutter",
        "react native",
        "swift",
        "kotlin"
    ],

    "DevOps": [
        "docker",
        "kubernetes",
        "devops",
        "linux",
        "ansible",
        "terraform"
    ],

    "Cloud": [
        "aws",
        "azure",
        "google cloud",
        "cloud computing"
    ],

    "Cybersecurity": [
        "security",
        "cybersecurity",
        "pentest",
        "ethical hacking",
        "информационная безопасность"
    ],

    "Testing": [
        "qa",
        "testing",
        "selenium",
        "pytest",
        "тестирование"
    ],

    "Databases": [
        "sql",
        "postgres",
        "postgresql",
        "mysql",
        "mongodb",
        "database",
        "база данных"
    ],

    "Business Analytics": [
        "power bi",
        "tableau",
        "business intelligence",
        "аналитика"
    ],

    "Project Management": [
        "scrum",
        "agile",
        "project management"
    ],

    "Design": [
        "figma",
        "ui",
        "ux",
        "web design",
        "graphic design"
    ],

    "Marketing": [
        "marketing",
        "seo",
        "smm",
        "digital marketing"
    ],

    "Finance": [
        "finance",
        "investment",
        "trading",
        "экономика"
    ],

    "Mathematics": [
        "mathematics",
        "math",
        "линейная алгебра",
        "теория вероятностей"
    ],

    "English": [
        "english",
        "английский язык",
        "ielts",
        "toefl"
    ]
}


def detect_category(text):
    text = text.lower()

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword in text:
                return category

    return "Other"


df["category"] = df["full_text"].apply(
    detect_category
)

df.to_csv(
    "data/courses_processed.csv",
    index=False
)

print(df["category"].value_counts())