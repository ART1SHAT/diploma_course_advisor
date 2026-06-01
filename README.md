# CourseAdvisor

Информационно-советующая система (ИС) для выбора образовательных программ. Проект разработан в рамках выпускной квалификационной работы (ВКР).

## Описание проекта

**CourseAdvisor** помогает пользователю подобрать курсы и программы обучения на основе формализованного профиля: бюджет, уровень знаний, доступное время, карьерная направленность, интересы и текстовая цель.

Система соответствует теме ВКР по следующим аспектам:

| Аспект ВКР | Реализация в проекте |
|------------|----------------------|
| Лингвистические переменные и нечёткий вывод | Модуль `fuzzy_engine.py`: термы (низкий/средний/высокий), правила R001–R005, трассировка активаций |
| Семантический поиск по естественному языку | Модуль `semantic_search.py`: эмбеддинги русскоязычной модели, ранжирование по цели и интересам |
| Гибридная рекомендательная модель | `recommender.py`: взвешенная формула семантики + нечёткого скоринга + бюджетного фактора |
| Объяснимость решений | `explain()`: блоки semantic / fuzzy_rules / budget, confidence, evidence, `fuzzy_trace` |
| Веб-интерфейс для демонстрации | FastAPI + Jinja2 (`templates/index.html`), сценарий «Демо» для защиты |

В базе — унифицированный каталог курсов (`data/unified_courses.json`, порядка 1300+ записей).

## Быстрый запуск

### Требования

- Python 3.10+
- ~2 ГБ свободного места (модель `sentence-transformers` загружается при первом запросе)

### Установка и запуск

```bash
cd diploma_course_advisor

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python demo_mvp.py
```

Откройте в браузере: **http://localhost:8000**

Проверка API без UI:

```bash
curl -s http://localhost:8000/api/health | python3 -m json.tool
```

Альтернативный запуск:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Первый вызов `/api/recommend` может занять 20–40 с: загрузка курсов, построение семантического индекса и калибровка нечётких термов.

## API эндпоинты

| Метод | Путь | Тело запроса | Ответ |
|-------|------|--------------|--------|
| `GET` | `/` | — | HTML-страница веб-интерфейса (`index.html`) |
| `GET` | `/api/health` | — | `{"status":"ok","components":[...],"templates_path":"...","templates_exists":true}` |
| `POST` | `/api/recommend` | JSON профиля (см. ниже) | `{"recommendations":[...],"explanations":{...},"meta":{...}}` |

### Тело `POST /api/recommend`

```json
{
  "budget": 50000,
  "knowledge_level": 3,
  "time_availability": 6,
  "career_focus": 1.0,
  "interests": ["python", "анализ данных"],
  "goals": "получить практические навыки для работы в IT"
}
```

| Поле | Тип | Описание |
|------|-----|----------|
| `budget` | `float \| null` | Бюджет в ₽, `≥ 0` |
| `knowledge_level` | `float \| null` | Уровень знаний, `0–10` |
| `time_availability` | `float \| null` | Часов в неделю, `≥ 0` |
| `career_focus` | `float \| null` | `0` — хобби, `0.5` — учёба, `1.0` — карьера |
| `interests` | `string[]` | Список интересов |
| `goals` | `string` | Текстовая цель обучения |

### Ответ `POST /api/recommend`

```json
{
  "recommendations": [
    {
      "id": "...",
      "title": "...",
      "price": 0,
      "format": "online",
      "skills": ["python"],
      "score": 0.742,
      "...": "..."
    }
  ],
  "explanations": {
    "<course_id>": {
      "course_id": "...",
      "confidence": 0.72,
      "evidence": "…фрагмент описания…",
      "semantic": [{ "match_type": "goal", "score": 0.81, "summary": "..." }],
      "fuzzy_rules": [{ "rule_id": "R001", "activation": 0.78, "summary": "..." }],
      "budget": { "within_budget": true, "summary": "..." },
      "explanations": ["✓ ..."],
      "fuzzy_trace": [{ "rule_id": "R001", "activation": 0.78, "details": [...], "conclusion": "..." }],
      "scores": { "semantic": 0.65, "fuzzy": 0.5 }
    }
  },
  "meta": {
    "fuzzy_rules_count": 5,
    "total_courses_in_db": 1310
  }
}
```

### Коды ошибок

| Код | Причина |
|-----|---------|
| `422` | Невалидные поля JSON (`detail` + `errors`) |
| `500` | Нет `data/unified_courses.json` или сбой рекомендателя |

## Как проводить демо (≈3 минуты)

Подробный тайминг — в файле [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

1. **Запустите сервер** (`python demo_mvp.py`), откройте http://localhost:8000.
2. **Проверьте health** (опционально): `curl http://localhost:8000/api/health` — статус `ok`, шаблоны на месте.
3. Нажмите **«Демо-сценарий»** — форма заполнится и отправится запрос (удобно, если мало времени на ввод).
4. Покажите **карточки рекомендаций**: процент соответствия, цена, формат, навыки.
5. Раскройте **блок «Объяснение»** (аккордеон): цель, правила, бюджет; укажите поле `confidence` и `evidence`.
6. Раскройте **«Трассировка правил»**: таблица Rule ID / Activation / Details / Conclusion — связь с § нечёткого вывода в ВКР.
7. При вопросе комиссии: формула гибридного скоринга на странице (`0.6 × семантика + 0.4 × нечёткий вывод` в подсказке UI; в коде — уточнённые веса в `recommender.py`).

## Архитектура

```
diploma_course_advisor/
├── demo_mvp.py              # Точка входа: uvicorn + app
├── app/
│   ├── main.py              # FastAPI factory, /, обработчики 422/500
│   ├── dependencies.py      # Ленивая загрузка HybridRecommender
│   ├── api/
│   │   ├── schemas.py       # Pydantic-модель RecommendationRequest
│   │   └── routes.py        # /api/recommend, /api/health
│   ├── user_profile.py      # UserProfile, to_fuzzy_input()
│   ├── fuzzy_engine.py      # Нечёткий вывод Mamdani, правила, get_trace()
│   ├── semantic_search.py   # SemanticIndex (sentence-transformers)
│   ├── recommender.py       # HybridRecommender: recommend(), explain()
│   └── course_loader.py     # Загрузка unified_courses.json
├── data/
│   └── unified_courses.json # Каталог курсов
└── templates/
    └── index.html           # Веб-UI для демонстрации
```

| Модуль | Ответственность |
|--------|-----------------|
| `demo_mvp.py` | Запуск сервера, проверка наличия шаблонов |
| `app/main.py` | Маршрутизация, Jinja2, глобальные exception handlers |
| `app/api/schemas.py` | Валидация входного JSON API |
| `app/api/routes.py` | HTTP-слой: профиль → рекомендации + объяснения |
| `app/dependencies.py` | Singleton рекомендателя, пути к данным |
| `app/user_profile.py` | Модель пользователя и маппинг в fuzzy-входы |
| `app/fuzzy_engine.py` | Лингвистические переменные, правила R001–R005, трассировка |
| `app/semantic_search.py` | Векторный индекс и поиск по цели/интересам |
| `app/recommender.py` | Гибридный скоринг, фильтры, структурированные explain |
| `app/course_loader.py` | Чтение JSON, статистика каталога |
| `templates/index.html` | Форма профиля, отображение результатов и трассировки |

Вспомогательные скрипты (`data/unify_pipeline.py`, `app/parser.py` и др.) используются для подготовки данных и **не требуются** для запуска демо MVP.

## Лицензия

См. файл [LICENSE](LICENSE).
