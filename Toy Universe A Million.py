import random
import networkx as nx
import matplotlib.pyplot as plt
import copy

MOD = 3  # Z_3 phase group

class Graph:
    # Internal graph representation; we convert to NetworkX for visualization
    def __init__(self):
        self.vertices = set()
        self.edges = {}  # (u, v) with u < v -> phase
        self.next_vertex_id = 0

    def add_vertex(self):
        v = self.next_vertex_id
        self.next_vertex_id += 1
        self.vertices.add(v)
        return v

    def add_edge(self, u, v, phase):
        if u == v:
            return
        a, b = sorted((u, v))
        self.edges[(a, b)] = phase % MOD

    def remove_edge(self, u, v):
        a, b = sorted((u, v))
        self.edges.pop((a, b), None)

    def get_phase(self, u, v):
        a, b = sorted((u, v))
        return self.edges.get((a, b))

    def copy(self):
        return copy.deepcopy(self)

    def constraint_measure(self):
        return len(self.vertices) + sum(self.edges.values())

    def __str__(self):
        edge_str = ", ".join(f"{u}-{v}:{p}" for (u, v), p in self.edges.items())
        return f"V={sorted(self.vertices)} | E={{ {edge_str} }}"


def seed_triangle():
    g = Graph()
    a = g.add_vertex()
    b = g.add_vertex()
    c = g.add_vertex()

    g.add_edge(a, b, 1)
    g.add_edge(b, c, 1)
    g.add_edge(c, a, 1)

    return g


def rule_subdivide_edge(g):
    if not g.edges:
        return None

    (u, v), phase = random.choice(list(g.edges.items()))

    new_g = g.copy()
    new_g.remove_edge(u, v)

    d = new_g.add_vertex()

    # choose a valid split a+b = phase mod 3
    splits = []
    for a in range(MOD):
        for b in range(MOD):
            if (a + b) % MOD == phase:
                splits.append((a, b))

    a, b = random.choice(splits)

    new_g.add_edge(u, d, a)
    new_g.add_edge(d, v, b)

    return new_g


def find_triangles(g):
    triangles = []
    verts = list(g.vertices)
    n = len(verts)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = verts[i], verts[j], verts[k]
                if (
                    g.get_phase(a, b) is not None
                    and g.get_phase(b, c) is not None
                    and g.get_phase(c, a) is not None
                ):
                    triangles.append((a, b, c))
    return triangles


def rule_triangle_flip(g):
    triangles = find_triangles(g)
    if not triangles:
        return None

    a, b, c = random.choice(triangles)
    new_g = g.copy()

    for (u, v) in [(a, b), (b, c), (c, a)]:
        phase = new_g.get_phase(u, v)
        new_g.add_edge(u, v, (phase + 1) % MOD)

    return new_g


def step(g, constraint_limit=2):
    candidates = []

    for rule in [rule_subdivide_edge, rule_triangle_flip]:
        new_g = rule(g)
        if new_g is None:
            continue

        if new_g.constraint_measure() <= g.constraint_measure() + constraint_limit:
            candidates.append(new_g)

    if not candidates:
        return g

    return random.choice(candidates)


def visualize(g, step_num, pause=1.5):
    plt.clf()

    G = nx.Graph()

    for v in g.vertices:
        G.add_node(v)

    edge_labels = {}
    for (u, v), p in g.edges.items():
        G.add_edge(u, v)
        edge_labels[(u, v)] = str(p)

    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=600)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(f"Toy Universe — Step {step_num}")
    plt.axis("off")
    plt.pause(pause)


def simulate(steps=1_000_000, pause=1.5, show_visual=True):
    g = seed_triangle()

    if show_visual:
        plt.figure(figsize=(6, 6))

    print("Initial:", g)
    if show_visual:
        visualize(g, 0, pause)

    for i in range(steps):
        g = step(g)
        print(f"Step {i+1}:", g)
        if show_visual:
            visualize(g, i+1, pause)


if __name__ == "__main__":
    simulate(steps=300)
