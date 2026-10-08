import numpy as np
import numpy.typing as npt


def stoer_wagner(weights: npt.NDArray[np.float64]) -> tuple[float, list[int]]:
    """
    Implementation of the Stoer-Wagner algorithm to find the minimum cut of a graph.

    Args:
    - weights (NDArray[float64]): symmetric weight matrix.

    Returns:
    Weight of the cut and the vertices on one side of the cut.
    """

    W = np.array(weights, dtype=np.float64)
    n = W.shape[0]

    active = list(range(n))
    groups = [[i] for i in range(n)]  # Super vertices.

    best_w, best_group = np.inf, []
    while len(active) > 1:
        w = np.zeros(n, dtype=np.float64)
        A = []
        candidates = set(active)

        # Building A.
        while candidates:
            v = max(candidates, key=lambda u: w[u])

            A.append(v)
            candidates.remove(v)
            for u in candidates:
                w[u] += W[v, u]

        # Checking to see if the current cut is the best one.
        s, t = A[-2], A[-1]
        if w[t] < best_w:
            best_w, best_group = float(w[t]), groups[t]

        # Merging t and s into a super vertex.
        groups[s] += groups[t]
        W[s, :] += W[t, :]
        W[:, s] += W[:, t]
        W[s, s] = 0
        active.remove(t)

    return best_w, best_group
