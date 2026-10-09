import numpy as np
import numpy.typing as npt


def _rp_simplex(
    T: npt.NDArray[np.float64],
    basis: list[int],
    allowed: npt.NDArray[np.bool],
    c_rp: npt.NDArray[np.float64],
    eps: float,
) -> tuple[np.float64, npt.NDArray[np.float64]]:
    """
    Implementation of the Simplex method for the RP problem. It assumes the both the tableau and the basis can be reused from the previous Simplex solution.

    # Note:
    The tableau (`T`) and the basis are modified in-place.

    # Args:
    - T (NDArray[float64]): tableau.
    - basis (list[int]): indices of the basic variables.
    - allowed (NDArray[numpy.bool]): variables to be considered in tableau.
    - c_rp (NDArray[float64]): reduced costs.
    - eps (float): tolerance.

    # Returns:
    Objective function value (`z_rp`) and variables `d`.
    """

    m = T.shape[0] - 1
    while True:
        # Finding entering variable.
        red = np.where(allowed, T[-1, :-1], 0.0)
        c_pivot = int(np.argmin(red))
        if red[c_pivot] >= -eps:
            break

        # Finding leaving variable.
        col, rhs = T[:-1, c_pivot], T[:-1, -1]
        mask = col > eps

        ratios = np.full(m, np.inf)
        ratios[mask] = rhs[mask] / col[mask]
        l_pivot = np.argmin(ratios)

        # Pivoting.
        T[l_pivot] /= T[l_pivot, c_pivot]

        f = T[:, c_pivot].copy()
        f[l_pivot] = 0
        rows = np.flatnonzero(f)
        T[rows] -= np.outer(f[rows], T[l_pivot])

        basis[l_pivot] = c_pivot

    # Compiling the results.
    z_rp = -T[-1, -1]
    d = c_rp[-m:] - T[-1, -1 - m : -1]
    return z_rp, d


def _cold_start(
    M: npt.NDArray[np.float64], N: int, m: int, n: int, b: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """
    Build a tableau from scratch for the RP problem.

    # Args:
    - M (NDArray[float64]): coefficient matrix for the constraints of the RP problem.
    - N (int): total number of variables (original + artificial).
    - m (int): number of constraints.
    - n (int): number of original variables.
    - b (NDArray[float64]): array containing the values on the right-hand side of the constraints.

    # Returns:
    Tableau created and the indices of the basic (artificial) variables.
    """

    T = np.zeros((m + 1, N + 1))
    T[:m, :-1], T[:m, -1] = M, b
    return T, list(range(n + m, N))


def primal_dual(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    y0: npt.NDArray[np.float64],
    eps: float = 1e-9,
    max_iter: int = 10000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], np.float64, int]:
    """
    Implementation of the primal-dual algorithm for problems in standard form. Thus, the function expects to receive parameters that already account for equality constraints and assume that slack variables have already been added (`A` and `c`).

    # Errors:
    Several checks are performed during preprocessing, and if any of them fail, the algorithm will raise a ValueError with its message. During the execution of the algorithm, a ValueError will be raised if the problem is infeasible and if the final solution does not match the dual solution.

    # Args:
    - A (NDArray[float64]): matrix containing the coefficients of the constraints.
    - b (NDArray[float64]): array containing the values on the right-hand side of the constraints.
    - c (NDArray[float64]): array containing the coefficients of the objective function.
    - y0 (NDArray[float64]): starting point for the dual problem.
    - eps (float): tolerance.
    - max_iter (int): maximum number of iterations.

    # Returns:
    Primal and dual problem variables (`x` and `y`), the value of the objective function (`z`), and the number of iterations (`iter`).
    """

    # Checking the feasibility of the parameters.
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")

    m, n = A.shape

    if b.shape != (m,):
        raise ValueError(f"b must be of size {m} (rows of A), but it has {b.shape}.")

    if c.shape != (n,):
        raise ValueError(f"c must be of size {n} (columns of A), but it has {c.shape}.")

    if y0.shape != (m,):
        raise ValueError(f"y0 must be of size {m}, but it has {y0.shape}.")

    if np.any(b < 0):
        raise ValueError("b must be >= 0.")

    if np.any(A.T @ y0 > c + eps):
        raise ValueError("y0 is not feasible for the dual: A^T y0 must be <= c.")

    # Building the tableau for the RP problem (all variables + artificials).
    N = n + 2 * m
    M = np.hstack([A, -np.eye(m), np.eye(m)])

    c_rp = np.concatenate([np.zeros(n), np.ones(2 * m)])
    T, basis = _cold_start(M, N, m, n, b)

    c_B = c_rp[basis]
    T[-1, :-1] = c_rp - c_B @ T[:-1, :-1]
    T[-1, -1] = -np.sum(b)

    # Algorithm.
    x = np.zeros(n)
    y = np.array(y0, dtype=np.float64)
    z = np.float64(np.inf)

    iter = 0
    unsolved = True
    while unsolved and iter < max_iter:
        # Selecting the variables to be included in tableau.
        r = np.maximum(c - A.T @ y, 0)
        J = np.flatnonzero(r <= eps)

        allowed = np.zeros(N, dtype=np.bool)
        allowed[J] = True
        allowed[n:] = True

        # The original basic column was removed from J.
        if any(g < n and not allowed[g] for g in basis):
            T, basis = _cold_start(M, N, m, n, b)

            c_B = c_rp[basis]
            T[-1, :-1] = c_rp - c_B @ T[:-1, :-1]
            T[-1, -1] = -np.sum(b)

        z_rp, d_rp = _rp_simplex(T, basis, allowed, c_rp, eps)

        if z_rp <= eps:
            # Optimal.
            x = np.zeros(n)
            for i, g in enumerate(basis):
                if g < n:
                    x[g] = T[i, -1]

            z = np.float64(c @ x)

            if not np.isclose(z, b @ y, atol=eps):
                raise ValueError("The primal solution differs from the dual solution.")

            unsolved = False
        else:
            # Updating y.
            Ad = A.T @ d_rp

            mask = Ad > eps
            mask[J] = False

            ratios = np.full(n, np.inf)
            ratios[mask] = r[mask] / Ad[mask]

            if np.all(np.isinf(ratios)):
                raise ValueError(
                    "The dual problem is unbounded, and therefore the primal problem is infeasible."
                )

            y += ratios.min() * d_rp

        iter += 1

    return x, y, z, iter
