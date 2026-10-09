from collections.abc import Mapping

import numpy as np

Model = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def build_adjacency_matrix(data: dict) -> np.ndarray:
    """Symmetric capacities; sum parallel edges and skip loops. IDs are 1-based."""
    n = data["num_nodes"]
    weights = np.zeros((n, n), dtype=np.float64)
    for u, v, capacity in data["arcs"]:
        if u != v:
            weights[u - 1, v - 1] += capacity
            weights[v - 1, u - 1] += capacity
    return weights


def build_min_cut(data: dict) -> Model:
    """Return (A, b, c, y0) for undirected s-t min cut.

    Columns: potentials, cut variables, two surplus blocks.
    Rows: two inequalities per edge, d_s = 0, d_t = 1."""
    n = data["num_nodes"]
    m = len(data["arcs"])
    commodity = next(iter(data["sources"]))
    source = data["sources"][commodity]
    sink = data["sinks"][commodity]

    A = np.zeros((2 * m + 2, n + 3 * m), dtype=np.float64)
    b = np.zeros(2 * m + 2, dtype=np.float64)
    c = np.zeros(n + 3 * m, dtype=np.float64)
    y0 = np.zeros(2 * m + 2, dtype=np.float64)

    for i, (u, v, capacity) in enumerate(data["arcs"]):
        # x_e - d_u + d_v - w1_e = 0.
        A[i, n + i] = 1.0
        A[i, u - 1] -= 1.0
        A[i, v - 1] += 1.0
        A[i, n + m + i] = -1.0

        # x_e + d_u - d_v - w2_e = 0.
        A[m + i, n + i] = 1.0
        A[m + i, u - 1] += 1.0
        A[m + i, v - 1] -= 1.0
        A[m + i, n + 2 * m + i] = -1.0
        c[n + i] = capacity

    A[2 * m, source - 1] = 1.0
    A[2 * m + 1, sink - 1] = 1.0
    b[2 * m + 1] = 1.0
    return A, b, c, y0


def build_max_flow(data: dict, prizes: Mapping[int, float]) -> Model:
    """Return (A, b, c, y0) for undirected multicommodity flow.

    Capacity is shared across commodities and directions.
    Minimize negative weighted sink inflow; original maximum is -z."""
    oriented_arcs, capacity_groups = _undirected_layout(data)
    commodities = sorted(data["sources"])
    n, m, k = data["num_nodes"], len(data["arcs"]), len(commodities)
    num_arcs = len(oriented_arcs)
    num_flows = k * num_arcs
    rows = m + k * (n - 2)
    A = np.zeros((rows, num_flows + m), dtype=np.float64)
    b = np.zeros(rows, dtype=np.float64)
    c = np.zeros(num_flows + m, dtype=np.float64)

    # Shared capacity: total flow + slack = capacity.
    for e, (_, _, capacity) in enumerate(data["arcs"]):
        b[e] = capacity
        A[e, num_flows + e] = 1.0
        for position in range(k):
            for arc_index in capacity_groups[e]:
                A[e, position * num_arcs + arc_index] += 1.0

    row = m
    for position, commodity in enumerate(commodities):
        source = data["sources"][commodity]
        sink = data["sinks"][commodity]
        offset = position * num_arcs

        # Minimize negative weighted sink inflow.
        for a, (_, v) in enumerate(oriented_arcs):
            if v == sink:
                c[offset + a] = -prizes[commodity]

        # Flow balance: outflow - inflow = 0.
        for node in range(1, n + 1):
            if node == source or node == sink:
                continue
            for a, (u, v) in enumerate(oriented_arcs):
                if u == node:
                    A[row, offset + a] += 1.0
                if v == node:
                    A[row, offset + a] -= 1.0
            row += 1

    y0 = _initial_flow_dual(prizes, m, rows)
    return A, b, c, y0


def _undirected_layout(data: dict) -> tuple[list[tuple[int, int]], list[list[int]]]:
    oriented_arcs = []
    capacity_groups = []
    for e, (u, v, _) in enumerate(data["arcs"]):
        oriented_arcs.append((u, v))
        oriented_arcs.append((v, u))
        capacity_groups.append([2 * e, 2 * e + 1])
    return oriented_arcs, capacity_groups


def _initial_flow_dual(
    prizes: Mapping[int, float], num_edges: int, num_rows: int
) -> np.ndarray:
    """Return a feasible initial dual point."""
    P = max(0.0, max(prizes.values()))
    y0 = np.zeros(num_rows, dtype=np.float64)
    y0[:num_edges] = -P
    return y0
