from collections.abc import Mapping

import numpy as np

Model = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def build_adjacency_matrix(data: dict) -> np.ndarray:
    """
    Builds a symmetric capacity matrix, summing parallel edges and ignoring loops.

    # Args:
    - data (dict): network with `num_nodes` and `arcs` (u, v, capacity). Node IDs are 1-based.

    # Returns:
    Capacity matrix `weights` (NDArray[float64]).
    """
    n = data["num_nodes"]
    weights = np.zeros((n, n), dtype=np.float64)
    for u, v, capacity in data["arcs"]:
        if u != v:
            weights[u - 1, v - 1] += capacity
            weights[v - 1, u - 1] += capacity
    return weights


def build_min_cut(data: dict) -> Model:
    """
    Builds the undirected minimum cut formulation for the first commodity.

    # Args:
    - data (dict): network with `num_nodes`, `arcs`, `sources`, and `sinks`. Node IDs are 1-based.

    # Returns:
    Constraint matrix `A`, right-hand side `b`, objective coefficients `c`,
    and initial dual vector `y0` (NDArray[float64]).
    """
    n = data["num_nodes"]
    m = len(data["arcs"])
    commodity = next(iter(data["sources"]))  # Get the first commodity ID
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


def build_max_flow(data: dict, prizes: Mapping[int, float] | None = None) -> Model:
    """
    Builds the undirected multicommodity flow formulation with shared capacities.

    # Args:
    - data (dict): network with `num_nodes`, `arcs`, `sources`, and `sinks`. Node IDs are 1-based.
    - prizes: optional commodity weights; defaults to 1 for all.

    # Returns:
    Constraint matrix `A`, right-hand side `b`, negated objective coefficients `c`,
    and initial dual vector `y0` (NDArray[float64]).
    """
    if prizes is None:
        prizes = {commodity: 1.0 for commodity in data["sources"]}
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
    for i, (_, _, capacity) in enumerate(data["arcs"]):
        b[i] = capacity
        A[i, num_flows + i] = 1.0
        for position in range(k):
            for arc_index in capacity_groups[i]:
                A[i, position * num_arcs + arc_index] += 1.0

    row = m
    for position, commodity in enumerate(commodities):
        source = data["sources"][commodity]
        sink = data["sinks"][commodity]
        offset = position * num_arcs
        premio = prizes[commodity]

        # Minimize negative weighted sink inflow.
        for i, (u, v) in enumerate(oriented_arcs):
            coluna = offset + i
            if v == sink:
                # Add prize to the objective for flow into the sink.
                c[coluna] -= premio

            if u == sink:
                # Add prize to the objective for flow out of the source.
                c[coluna] += premio

        # Flow balance: outflow - inflow = 0.
        for node in range(1, n + 1):
            if node == source or node == sink:
                continue
            for i, (u, v) in enumerate(oriented_arcs):
                if u == node:
                    A[row, offset + i] += 1.0
                if v == node:
                    A[row, offset + i] -= 1.0
            row += 1

    y0 = _initial_flow_dual(prizes, m, rows)
    return A, b, c, y0


def _undirected_layout(data: dict) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """
    Expands undirected edges into pairs of directed arcs.

    # Args:
    - data (dict): network with `arcs` (u, v, capacity).

    # Returns:
    Directed arcs `oriented_arcs` and arc indices grouped by edge `capacity_groups`.
    """
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
    """
    Builds the initial dual vector for the flow formulation.

    # Args:
    - prizes (Mapping[int, float]): nonempty mapping of commodity weights.
    - num_edges (int): number of capacity constraints.
    - num_rows (int): total number of constraints.

    # Returns:
    Dual vector `y0` (NDArray[float64]), with capacity entries set to
    -max(0, max(prizes.values())) and remaining entries set to zero.
    """
    P = max(0.0, max(prizes.values()))
    y0 = np.zeros(num_rows, dtype=np.float64)
    y0[:num_edges] = -P
    return y0
