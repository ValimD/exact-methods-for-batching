import numpy as np
import numpy.typing as npt


def _active_restrictions(
    A: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    eps: float = 1e-9
) -> npt.NDArray[np.intp]:
    """
    Checks which dual restrictions are active and returns an index of them.

    # Args:
    - A (NDArray[float64]): matrix containing the coefficients of the constraints.
    - c (NDArray[float64]): array containing the coefficients of the objective function.
    - y (NDArray[float64]): solution to the dual problem.
    - eps (float): tolerance.

    # Returns:
    Index of restrictions (`J`).
    """

    r = c - A.T @ y
    return np.flatnonzero(np.isclose(r, 0, atol=eps))


def _build_rp(
    A: npt.NDArray[np.float64],
    J: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Construct the matrix of constraint coefficients and the objective function coefficients for the RP problem.

    # Args:
    - A (NDArray[float64]): matrix containing the coefficients of the constraints of the original problem.
    - J (NDArray[intp]): array containing the indices of the columns to be considered in the RP problem.

    # Returns:
    Matrix of constraint coefficients (`A_rp`) and the objective function coefficients (`c_rp`).
    """

    m, _ = A.shape
    A_j = A[:, J]
    return np.hstack([A_j, np.eye(m)]), np.concatenate([np.zeros(len(J)), np.ones(m)])


def _simplex(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    eps: float = 1e-9,
) -> tuple[npt.NDArray[np.float64], np.float64, npt.NDArray[np.float64]]:
    """
    Implementation of the Simplex algorithm for problems in standard form. As a result, the algorithm expects matrix `A` and the constants `c` to already be in the appropriate dimensions, taking into account the slack variables for inequality constraints.

    Furthermore, since the algorithm was designed to be used only with the primal-dual method, it also assumes that the last `m` columns of `A` form an identity matrix, and that b >= 0.

    # Args:
    - A (NDArray[float64]): matrix containing the coefficients of the constraints.
    - b (NDArray[float64]): array containing the values on the right-hand side of the constraints.
    - c (NDArray[float64]): array containing the coefficients of the objective function.
    - eps (float): tolerance.

    # Returns:
    The values of the problem variables (`x`), the value of the objective function (`z`), and the values of the dual variables (`d`).
    """

    tableau = np.hstack([A, b.reshape(-1, 1)], dtype=np.float64)

    B_indexes = list(range(A.shape[1]))[-A.shape[0] :]
    c_B = c[B_indexes]
    fo = np.hstack([c - c_B @ A, -(c_B @ b)], dtype=np.float64)

    tableau = np.vstack([tableau, fo])

    while np.any(tableau[-1, :-1] < -eps):
        # Finding entering and leaving variables.
        c_pivot = np.argmax(tableau[-1, :-1] < -eps)

        column = tableau[:-1, c_pivot]
        rhs = tableau[:-1, -1]

        ratios = np.full_like(rhs, np.inf)
        mask = column > eps
        ratios[mask] = rhs[mask] / column[mask]

        l_pivot = np.argmin(ratios)

        # Pivoting.
        tableau[l_pivot] /= tableau[l_pivot, c_pivot]

        a = tableau[:, c_pivot].copy()
        a[l_pivot] = 0
        b = tableau[l_pivot].copy()

        tableau -= np.outer(a, b)

        B_indexes[l_pivot] = int(c_pivot)

    # Compiling the results.
    x = np.zeros(A.shape[1])
    for i, bv in enumerate(B_indexes):
        x[bv] = tableau[i, -1]

    z = -tableau[-1, -1]

    # Compiling the dual results.
    m = A.shape[0]
    reduced_costs = tableau[-1, :-1]
    d = c[-m:] - reduced_costs[-m:]

    return x, z, d


def primal_dual(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    y0: npt.NDArray[np.float64],
    eps: float = 1e-9,
    max_iter: int = 100
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], np.float64, int]:
    """
    Implementation of the primal-dual algorithm for problems in standard form. Thus, the function expects to receive parameters that already account for equality constraints and assume that slack variables have already been added (`A` and `c`).

    # Errors
    Several checks are performed during preprocessing, and if any of them fail, the algorithm will raise a ValueError with its message. During the execution of the algorithm, a ValueError will be raised if the problem is infeasible and if the final solution does not match the dual solution.

    # Args
    - A (NDArray[float64]): matrix containing the coefficients of the constraints.
    - b (NDArray[float64]): array containing the values on the right-hand side of the constraints.
    - c (NDArray[float64]): array containing the coefficients of the objective function.
    - y0 (NDArray[float64]): starting point for the dual problem.
    - eps (float): tolerance.
    - max_iter (int): maximum number of iterations.

    # Returns
    Primal and dual problem variables (`x` and `y`), the value of the objective function (`z`), and the number of iterations (`iter`). 
    """

    # Checking the feasibility of the parameters.
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")

    num_constraints, num_variables = A.shape

    if b.shape != (num_constraints,):
        raise ValueError(f"b must be of size {num_constraints} (rows of A), but it has {b.shape}.")

    if c.shape != (num_variables,):
        raise ValueError(f"c must be of size {num_variables} (columns of A), but it has {c.shape}.")

    if y0.shape != (num_constraints,):
        raise ValueError(f"y0 must be of size {num_constraints}, but it has {y0.shape}.")

    if np.any(b < 0):
        raise ValueError("b must be >= 0.")

    if np.any(A.T @ y0 > c + eps):
        raise ValueError("y0 is not feasible for the dual: A^T y0 must be <= c.")

    # Algorithm.
    x = np.zeros(num_variables)
    y = np.array(y0, dtype = np.float64)
    z = np.float64(np.inf)

    iter = 0
    unsolved = True
    while unsolved and iter < max_iter:
        # Setting Up the RP Problem.
        J = _active_restrictions(A, c, y, eps)
        A_rp, c_rp = _build_rp(A, J)
        x_rp, z_rp, d_rp = _simplex(A_rp, b, c_rp, eps)

        if np.isclose(z_rp, 0.0, atol = eps):
            # Optimal.
            x[J] = x_rp[: len(J)]
            z = np.float64(c @ x)

            if not np.isclose(z, b @ y, atol = eps):
                raise ValueError("The primal solution differs from the dual solution.")
            
            unsolved = False
        else:
            # Updating y.
            Ad = A.T @ d_rp

            mask = Ad > eps
            mask[J] = False

            r = np.maximum(c - A.T @ y, 0)

            ratios = np.full(num_variables, np.inf)
            ratios[mask] = r[mask] / Ad[mask]

            if np.all(np.isinf(ratios)):
                raise ValueError("The dual problem is unbounded, and therefore the primal problem is infeasible.")

            y += ratios.min() * d_rp

        iter += 1

    return x, y, z, iter