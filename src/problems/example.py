import numpy as np
import numpy.typing as npt


def example_1() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Returns the values needed for the primal-dual algorithm for the following optimization problem:

    min 2x + 3y

    s.t.

        x + y >= 5

        2x + y >= 7

        x, y >= 0

    # Returns:
    Matrix containing the coefficients of the constraints (`A`), array containing the values on the right-hand side of the constraints (`b`), array containing the coefficients of the objective function (`c`), starting point for the dual problem (`y0`).
    """

    A = np.array([[1, 1, -1, 0], [2, 1, 0, -1]], dtype = np.float64)
    b = np.array([5, 7], dtype = np.float64)
    c = np.array([2, 3, 0, 0], dtype = np.float64)
    y = np.array([0, 0], dtype = np.float64)
    return A, b, c, y

def example_2() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Returns the values needed for the primal-dual algorithm for the following optimization problem:

    min x + 3y

    s.t.

        x + 2y >= 2

        2x + y >= 2

        x, y >= 0

    # Returns:
    Matrix containing the coefficients of the constraints (`A`), array containing the values on the right-hand side of the constraints (`b`), array containing the coefficients of the objective function (`c`), starting point for the dual problem (`y0`).
    """

    A = np.array([[1, 2, -1, 0], [2, 1, 0, -1]], dtype = np.float64)
    b = np.array([2, 2], dtype = np.float64)
    c = np.array([1, 3, 0, 0], dtype = np.float64)
    y = np.array([0, 0], dtype = np.float64)
    return A, b, c, y