import numpy as np
import numpy.typing as npt
import pulp


def build_problem(
    A: npt.NDArray[np.float64], b: npt.NDArray[np.float64], c: npt.NDArray[np.float64]
) -> pulp.LpProblem:
    """
    Use PuLP to construct a minimization problem with equality constraints. To do this, the input variables must be in standard form.

    Args:
    - A (NDArray[float64]): matrix containing the coefficients of the constraints.
    - b (NDArray[float64]): array containing the values on the right-hand side of the constraints.
    - c (NDArray[float64]): array containing the coefficients of the objective function.

    Returns:
    Instance of the modeled problem.
    """
    
    m, n = A.shape

    # Building problem.
    prob = pulp.LpProblem("Optimization_Problem", pulp.LpMinimize)

    # Creating variables and adding objective function to problem.
    x = [prob.add_variable(f"x_{i}", lowBound=0) for i in range(n)]
    prob += (pulp.lpSum(c[i] * x[i] for i in range(n)), "Objective_Function")

    # Adding restrictions.
    for i in range(m):
        prob += (pulp.lpSum(A[i, j] * x[j] for j in range(n)) == b[i], f"Restricao_{i}")

    return prob
