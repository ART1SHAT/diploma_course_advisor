"""
Граф компетенций: курс → навык → профессия (§1.1, §2.1).
Построение и обход через NetworkX (BFS, кратчайший путь).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ONTOLOGY_PATH = PROJECT_ROOT / "data" / "ontology" / "competency_graph.json"

NODE_COURSE = "course"
NODE_SKILL = "skill"
NODE_PROFESSION = "profession"


def _node_id(kind: str, name: str) -> str:
    return f"{kind}:{name}"


def _parse_node_id(node: str) -> Tuple[str, str]:
    kind, _, name = node.partition(":")
    return kind, name


class CompetencyGraph:
    """Онтология навыков и связей курс → навык → профессия."""

    def __init__(self, ontology_path: Optional[Path] = None) -> None:
        self.ontology_path = Path(ontology_path or DEFAULT_ONTOLOGY_PATH)
        self._raw: Dict[str, Any] = {}
        self.graph: nx.DiGraph = nx.DiGraph()

    def load(self) -> None:
        """Загружает JSON-онтологию и строит направленный граф."""
        if not self.ontology_path.is_file():
            raise FileNotFoundError(
                f"Онтология графа компетенций не найдена: {self.ontology_path}"
            )

        with open(self.ontology_path, encoding="utf-8") as f:
            self._raw = json.load(f)

        g = nx.DiGraph()
        skills_map: Dict[str, List[str]] = self._raw.get("skills", {})
        courses_map: Dict[str, Dict[str, List[str]]] = self._raw.get("courses", {})

        for skill, professions in skills_map.items():
            skill_nid = _node_id(NODE_SKILL, skill)
            g.add_node(skill_nid, kind=NODE_SKILL, label=skill)
            for prof in professions:
                prof_nid = _node_id(NODE_PROFESSION, prof)
                g.add_node(prof_nid, kind=NODE_PROFESSION, label=prof)
                g.add_edge(skill_nid, prof_nid, relation="skill_to_profession")

        for course_id, meta in courses_map.items():
            course_nid = _node_id(NODE_COURSE, course_id)
            g.add_node(course_nid, kind=NODE_COURSE, label=course_id)
            for skill in meta.get("skills", []):
                skill_nid = _node_id(NODE_SKILL, skill)
                if skill_nid not in g:
                    g.add_node(skill_nid, kind=NODE_SKILL, label=skill)
                g.add_edge(course_nid, skill_nid, relation="course_to_skill")
            # Профессии курса — метаданные для API; в графе только skill → profession
            for prof in meta.get("professions", []):
                prof_nid = _node_id(NODE_PROFESSION, prof)
                if prof_nid not in g:
                    g.add_node(prof_nid, kind=NODE_PROFESSION, label=prof)

        self.graph = g

    def course_exists(self, course_id: str) -> bool:
        return course_id in self._raw.get("courses", {})

    def get_course_meta(self, course_id: str) -> Dict[str, List[str]]:
        courses = self._raw.get("courses", {})
        if course_id not in courses:
            raise KeyError(f"Курс {course_id} отсутствует в онтологии графа")
        return courses[course_id]

    def get_path(
        self, course_id: str, profession: str
    ) -> List[str]:
        """
        Кратчайший путь от курса к профессии (через навыки при необходимости).
        Возвращает список идентификаторов узлов графа.
        """
        start = _node_id(NODE_COURSE, course_id)
        end = _node_id(NODE_PROFESSION, profession)
        if start not in self.graph:
            raise KeyError(f"Курс {course_id} не найден в графе")
        if end not in self.graph:
            raise KeyError(f"Профессия {profession} не найдена в графе")
        try:
            return nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return [start, end]

    def get_related_skills(self, skill: str, max_depth: int = 2) -> List[str]:
        """
        Навыки, связанные с данным: общие профессии и соседи по BFS (без профессий).
        """
        start = _node_id(NODE_SKILL, skill)
        if start not in self.graph:
            return []

        related: Set[str] = set()
        professions = {
            _parse_node_id(n)[1]
            for n in self.graph.successors(start)
            if _parse_node_id(n)[0] == NODE_PROFESSION
        }

        for other_skill, profs in self._raw.get("skills", {}).items():
            if other_skill == skill:
                continue
            if professions & set(profs):
                related.add(other_skill)

        # BFS по графу: соседние навыки через общие профессии
        visited = {start}
        frontier = [start]
        depth = 0
        while frontier and depth < max_depth:
            next_frontier: List[str] = []
            for node in frontier:
                for neighbor in self.graph.successors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    kind, name = _parse_node_id(neighbor)
                    if kind == NODE_SKILL and name != skill:
                        related.add(name)
                    if kind != NODE_COURSE:
                        next_frontier.append(neighbor)
                for neighbor in self.graph.predecessors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    kind, name = _parse_node_id(neighbor)
                    if kind == NODE_SKILL and name != skill:
                        related.add(name)
                    if kind != NODE_COURSE:
                        next_frontier.append(neighbor)
            frontier = next_frontier
            depth += 1

        return sorted(related)

    def format_explanation_path(self, path_nodes: List[str]) -> str:
        """Человекочитаемая цепочка: Курс → Навык → Профессия."""
        labels: List[str] = []
        kind_labels = {
            NODE_COURSE: "Курс",
            NODE_SKILL: "Навык",
            NODE_PROFESSION: "Профессия",
        }
        for node in path_nodes:
            kind, name = _parse_node_id(node)
            prefix = kind_labels.get(kind, kind)
            labels.append(f"{prefix}({name})")
        return " → ".join(labels)

    def build_api_payload(
        self,
        course_id: str,
        profession: Optional[str] = None,
        include_related: bool = True,
    ) -> Dict[str, Any]:
        """Узлы и рёбра подсграфа для визуализации и explanation_path."""
        meta = self.get_course_meta(course_id)
        target_profession = profession or (meta.get("professions") or [None])[0]
        if not target_profession:
            raise ValueError(f"Для курса {course_id} не задана целевая профессия")

        path_nodes = self.get_path(course_id, target_profession)
        node_ids: Set[str] = set(path_nodes)

        if include_related:
            for skill in meta.get("skills", []):
                node_ids.add(_node_id(NODE_SKILL, skill))
                for rel in self.get_related_skills(skill):
                    node_ids.add(_node_id(NODE_SKILL, rel))

        nodes: List[Dict[str, str]] = []
        for nid in sorted(node_ids):
            if nid not in self.graph:
                continue
            attrs = self.graph.nodes[nid]
            kind, name = _parse_node_id(nid)
            nodes.append(
                {
                    "id": nid,
                    "label": attrs.get("label", name),
                    "type": kind,
                }
            )

        edges: List[Dict[str, str]] = []
        seen_edges: Set[Tuple[str, str]] = set()
        for u, v, data in self.graph.edges(data=True):
            if u in node_ids and v in node_ids:
                key = (u, v)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(
                    {
                        "source": u,
                        "target": v,
                        "relation": data.get("relation", ""),
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "explanation_path": self.format_explanation_path(path_nodes),
            "course_id": course_id,
            "profession": target_profession,
        }


_graph_instance: Optional[CompetencyGraph] = None


def get_competency_graph() -> CompetencyGraph:
    """Ленивая загрузка синглтона графа компетенций."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = CompetencyGraph()
        _graph_instance.load()
    return _graph_instance


def reset_competency_graph() -> None:
    """Сброс кэша (для тестов)."""
    global _graph_instance
    _graph_instance = None
