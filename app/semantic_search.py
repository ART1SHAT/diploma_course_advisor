# app/semantic_search.py — MINIMAL & BULLETPROOF
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

def _ensure_ndarray(data, expected_shape_2d: bool = True) -> Optional[np.ndarray]:
    """Гарантирует: возвращает только np.ndarray или None"""
    if data is None:
        return None
    # Если уже ndarray — проверяем тип и возвращаем
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    # Пробуем конвертировать
    try:
        arr = np.asarray(data, dtype=np.float32)
        # Если ожидали 2D, а получили 1D — добавляем измерение
        if expected_shape_2d and arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception as e:
        logger.error(f"⚠️ Не удалось конвертировать в ndarray: {e}")
        return None

class SemanticIndex:
    """Семантический поиск: минимализм + защита"""
    
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        self.model_name = model_name
        self.model = None
        self.vectors: Optional[np.ndarray] = None
        self.courses: List[Dict] = []
        self._ready = False

    def build(self, courses: List[Dict]):
        """Индексирует курсы — с полной защитой"""
        self.courses = courses if courses else []
        self._ready = False
        
        if not self.courses:
            logger.warning("⚠️ Нет курсов для индексации")
            return
            
        # Ленивая загрузка модели
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"✅ Модель {self.model_name} загружена")
            except Exception as e:
                logger.error(f"⚠️ Не удалось загрузить модель: {e}")
                self._ready = True  # активируем fallback
                return
        
        try:
            # Подготовка текстов
            texts = []
            for c in self.courses:
                t = f"{c.get('title','')} {c.get('description','')[:800]} {' '.join(c.get('competencies',[]))}".strip()
                texts.append(t)
            
            # Кодирование — БЕЗ dtype, БЕЗ show_progress_bar
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            
            # ❗ Ключевое: жёсткая конвертация в массив
            self.vectors = _ensure_ndarray(embeddings, expected_shape_2d=True)
            
            if self.vectors is not None and self.vectors.size > 0:
                logger.info(f"✅ Индекс готов: {self.vectors.shape}")
                self._ready = True
            else:
                logger.warning("⚠️ Векторы пустые, используем fallback")
                self._ready = True  # fallback активен
                
        except Exception as e:
            logger.error(f"⚠️ Ошибка индексации: {e}", exc_info=True)
            self.vectors = None
            self._ready = True  # fallback

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Dict, float]]:
        """Поиск — с полной защитой"""
        if not self.courses:
            return []
        
        # Fallback если нет векторов
        if not self._ready or self.vectors is None or self.vectors.size == 0:
            return self._keyword_fallback(query, top_k)
        
        try:
            # Кодирование запроса
            q_emb = self.model.encode([query], normalize_embeddings=True)
            q_vec = _ensure_ndarray(q_emb, expected_shape_2d=False)
            
            if q_vec is None or q_vec.size == 0:
                return self._keyword_fallback(query, top_k)
            
            # Убираем лишнюю размерность если есть
            if q_vec.ndim == 2 and q_vec.shape[0] == 1:
                q_vec = q_vec[0]
            
            # ❗ Проверка: vectors должен быть ndarray
            if not isinstance(self.vectors, np.ndarray):
                logger.warning("⚠️ self.vectors не ndarray, конвертируем")
                self.vectors = _ensure_ndarray(self.vectors, expected_shape_2d=True)
                if self.vectors is None:
                    return self._keyword_fallback(query, top_k)
            
            # Косинусная близость
            sims = np.dot(self.vectors, q_vec)
            
            # Безопасный топ-k
            n = len(sims)
            k = min(top_k, n)
            if k <= 0:
                return []
            
            top_idx = np.argsort(sims)[::-1][:k]
            
            results = []
            for idx in top_idx:
                if 0 <= idx < len(self.courses):
                    score = max(0.0, min(1.0, (float(sims[idx]) + 1) / 2))
                    results.append((self.courses[idx], score))
            return results
            
        except Exception as e:
            logger.error(f"⚠️ Ошибка поиска: {e}", exc_info=True)
            return self._keyword_fallback(query, top_k)

    def _keyword_fallback(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Простой поиск по ключевым словам"""
        if not self.courses or not query:
            return [(c, 0.5) for c in self.courses[:min(top_k, len(self.courses))]]
        
        q_words = set(w.lower() for w in query.split() if len(w) > 2)
        results = []
        
        for c in self.courses:
            text = f"{c.get('title','')} {c.get('description','')} {' '.join(c.get('competencies',[]))}".lower()
            if not q_words:
                score = 0.5
            else:
                matches = sum(1 for w in q_words if w in text)
                score = matches / len(q_words)
            results.append((c, float(min(score, 1.0))))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]