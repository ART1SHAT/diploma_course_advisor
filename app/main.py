from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Course Advisor")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <h1>Информационно-советующая система</h1>
    <p>Система выбора образовательной программы</p>
    """