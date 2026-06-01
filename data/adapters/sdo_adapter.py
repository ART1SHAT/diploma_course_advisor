# data/adapters/sdo_adapter.py
import requests
from bs4 import BeautifulSoup
import json
import time
import random
from typing import List, Dict
from data.models import UnifiedCourse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDOAdapter:
    """Адаптер для загрузки курсов с платформы СЦОС (platformaedu.ru)"""
    
    BASE_URL = "https://platformaedu.ru"
    CATALOG_URL = f"{BASE_URL}/courses"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9"
    }

    def fetch_catalog(self, max_pages: int = 3) -> List[Dict]:
        """Скачивает превью курсов из каталога"""
        courses = []
        
        for page in range(1, max_pages + 1):
            url = f"{self.CATALOG_URL}?page={page}"
            try:
                logger.info(f"📥 Загрузка страницы {page}: {url}")
                response = requests.get(url, headers=self.HEADERS, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Парсинг карточек курсов (селекторы могут меняться, адаптируем под типичную структуру)
                # Ищем ссылки на курсы, которые содержат /courses/ в href
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link['href']
                    if '/courses/' in href and href != '/courses/':
                        # Пропускаем дубли и некорректные ссылки
                        course_id = href.strip('/').split('/')[-1]
                        if not course_id.isdigit() and len(course_id) > 2:
                            title = link.get_text(strip=True)
                            if title and len(title) > 5:
                                courses.append({
                                    "id": f"sdo_{course_id}",
                                    "title": title,
                                    "url": f"{self.BASE_URL}{href}",
                                    "source": "СЦОС"
                                })
                
                # Дедупликация по URL
                seen = set()
                unique_courses = []
                for c in courses:
                    if c['url'] not in seen:
                        seen.add(c['url'])
                        unique_courses.append(c)
                courses = unique_courses
                
                logger.info(f"✅ Найдено {len(courses)} уникальных курсов на странице {page}")
                time.sleep(random.uniform(1, 3)) # Анти-спам
                
            except Exception as e:
                logger.error(f"❌ Ошибка при загрузке страницы {page}: {e}")
                
        return courses

    def enrich_courses(self, raw_courses: List[Dict]) -> List[UnifiedCourse]:
        """Превращает сырые данные в UnifiedCourse"""
        result = []
        for raw in raw_courses:
            try:
                # СЦОС курсы обычно бесплатные, но могут быть платные
                # Для MVP ставим 0 или берем из описания, если парсим детальную страницу
                course = UnifiedCourse(
                    id=raw['id'],
                    title=raw['title'],
                    provider="СЦОС",
                    provider_type="government_platform",
                    price=0.0,
                    format="online",
                    language="ru",
                    level=["beginner", "intermediate"],
                    competencies=["образование", "цифровые навыки"], # Заглушка, улучшается при парсинге описания
                    certification="state",
                    url=raw['url'],
                    description=f"Курс с платформы непрерывного образования СЦОС. {raw['title']}",
                    source_quality=0.95
                )
                result.append(course)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка валидации курса {raw['id']}: {e}")
        
        return result

    def run(self, save_path: str = "data/sdo_courses.json") -> List[UnifiedCourse]:
        """Полный пайплайн: сбор -> обогащение -> сохранение"""
        logger.info("🚀 Запуск адаптера СЦОС...")
        raw = self.fetch_catalog(max_pages=2) # Ограничиваем 2 страницами для скорости MVP
        if not raw:
            logger.error("⛔ Не удалось получить данные. Проверьте доступ к platformaedu.ru")
            return []
            
        courses = self.enrich_courses(raw)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump([c.dict() for c in courses], f, ensure_ascii=False, indent=2)
            
        logger.info(f"💾 Сохранено {len(courses)} курсов СЦОС в {save_path}")
        return courses

if __name__ == "__main__":
    adapter = SDOAdapter()
    adapter.run()