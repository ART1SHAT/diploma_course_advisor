"""Тесты графа компетенций и GET /api/graph/{course_id} (§1.1, §2.1)."""
import json
import tempfile
import unittest
from pathlib import Path

from app.services.competency_graph import (
    CompetencyGraph,
    _node_id,
    get_competency_graph,
    reset_competency_graph,
)


SAMPLE_ONTOLOGY = {
    "skills": {
        "python": ["data_analyst", "backend_dev"],
        "pandas": ["data_analyst"],
        "sql": ["data_analyst", "backend_dev"],
    },
    "courses": {
        "stepik_289612": {
            "skills": ["python"],
            "professions": ["data_analyst", "backend_dev"],
        },
        "demo_data_analytics": {
            "skills": ["python", "pandas", "sql"],
            "professions": ["data_analyst"],
        },
    },
}


class TestCompetencyGraph(unittest.TestCase):
    def setUp(self):
        reset_competency_graph()
        self.tmp = tempfile.TemporaryDirectory()
        self.ontology_path = Path(self.tmp.name) / "competency_graph.json"
        self.ontology_path.write_text(
            json.dumps(SAMPLE_ONTOLOGY, ensure_ascii=False), encoding="utf-8"
        )
        self.graph = CompetencyGraph(self.ontology_path)
        self.graph.load()

    def tearDown(self):
        reset_competency_graph()
        self.tmp.cleanup()

    def test_load_builds_nodes_and_edges(self):
        self.assertGreater(self.graph.graph.number_of_nodes(), 0)
        self.assertGreater(self.graph.graph.number_of_edges(), 0)
        self.assertIn(_node_id("course", "stepik_289612"), self.graph.graph)

    def test_get_path_course_to_profession(self):
        path = self.graph.get_path("stepik_289612", "data_analyst")
        self.assertEqual(path[0], _node_id("course", "stepik_289612"))
        self.assertEqual(path[-1], _node_id("profession", "data_analyst"))
        self.assertIn(_node_id("skill", "python"), path)

    def test_get_related_skills_shares_profession(self):
        related = self.graph.get_related_skills("python")
        self.assertIn("pandas", related)
        self.assertIn("sql", related)

    def test_format_explanation_path(self):
        path = self.graph.get_path("stepik_289612", "backend_dev")
        text = self.graph.format_explanation_path(path)
        self.assertIn("Курс", text)
        self.assertIn("Навык", text)
        self.assertIn("Профессия", text)
        self.assertIn("→", text)

    def test_build_api_payload(self):
        payload = self.graph.build_api_payload("stepik_289612", profession="data_analyst")
        self.assertTrue(payload["nodes"])
        self.assertTrue(payload["edges"])
        self.assertIn("Курс", payload["explanation_path"])
        types = {n["type"] for n in payload["nodes"]}
        self.assertIn("course", types)
        self.assertIn("skill", types)


class TestGraphApi(unittest.TestCase):
    def setUp(self):
        reset_competency_graph()

    def tearDown(self):
        reset_competency_graph()

    def test_get_graph_endpoint(self):
        from fastapi.testclient import TestClient

        from app.main import create_app

        client = TestClient(create_app())
        resp = client.get("/api/graph/stepik_289612")
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("explanation_path", data)
        self.assertIn("Курс", data["explanation_path"])
        self.assertEqual(data["course_id"], "stepik_289612")

    def test_get_graph_unknown_course_404(self):
        from fastapi.testclient import TestClient

        from app.main import create_app

        client = TestClient(create_app())
        resp = client.get("/api/graph/unknown_course_xyz")
        self.assertEqual(resp.status_code, 404)

    def test_get_graph_with_profession_query(self):
        from fastapi.testclient import TestClient

        from app.main import create_app

        client = TestClient(create_app())
        resp = client.get(
            "/api/graph/stepik_289612", params={"profession": "backend_dev"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["profession"], "backend_dev")
        self.assertIn("backend_dev", resp.json()["explanation_path"])


class TestCompetencyGraphSingleton(unittest.TestCase):
    def tearDown(self):
        reset_competency_graph()

    def test_singleton_loads_project_ontology(self):
        g = get_competency_graph()
        self.assertTrue(g.course_exists("stepik_289612"))


if __name__ == "__main__":
    unittest.main()
