#!/usr/bin/env python3
"""Генерация assets/competency_graph_sample.png для презентации (§2.1)."""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from app.services.competency_graph import get_competency_graph, reset_competency_graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "assets" / "competency_graph_sample.png"

NODE_COLORS = {
    "course": "#2563eb",
    "skill": "#16a34a",
    "profession": "#dc2626",
}


def main() -> None:
    reset_competency_graph()
    cg = get_competency_graph()
    payload = cg.build_api_payload("stepik_289612", profession="data_analyst")

    g = nx.DiGraph()
    for node in payload["nodes"]:
        g.add_node(node["id"], label=node["label"], type=node["type"])
    for edge in payload["edges"]:
        g.add_edge(edge["source"], edge["target"])

    pos = nx.spring_layout(g, seed=42, k=1.4)
    colors = [NODE_COLORS.get(g.nodes[n].get("type", ""), "#64748b") for n in g.nodes]

    plt.figure(figsize=(11, 7))
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=16, edge_color="#94a3b8")
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=1400, alpha=0.92)
    labels = {n: g.nodes[n].get("label", n) for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_weight="bold")

    plt.title(
        "Граф компетенций: курс → навык → профессия\n"
        f"{payload['explanation_path']}",
        fontsize=12,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {OUTPUT}")


if __name__ == "__main__":
    main()
