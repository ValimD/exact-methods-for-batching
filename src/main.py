import argparse
import sys
from time import perf_counter

import numpy as np
import pulp

from dataset_reader import dataset_reader
from methods.primal_dual import primal_dual
from methods.pulp import build_problem
from methods.stoer_wagner import stoer_wagner
from problems.example import example_1, example_2
from problems.network_formulations import (
    build_adjacency_matrix,
    build_max_flow,
    build_min_cut,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph optimization models.")

    parser.add_argument(
        "problem",
        type=int,
        choices=[0, 1, 2],
        help="Problem to solve: 0 (Examples), 1 (Min Cut), 2 (Max Flow).",
    )

    parser.add_argument(
        "instance_path", nargs="?", help="Path to the problem instance."
    )

    parser.add_argument(
        "--prizes",
        type=float,
        nargs="+",
        help="P2 prizes by ascending commodity ID (default: all 1).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Primal-dual iteration limit."
    )
    args = parser.parse_args()

    if args.problem != 0 and args.instance_path is None:
        parser.error("instance_path is required for problems 1 and 2.")
    if args.prizes is not None and args.problem != 2:
        parser.error("--prizes requires problem 2.")
    if args.max_iter < 1:
        parser.error("--max-iter must be positive.")

    return args


def main(problem: int, instance_path: str | None, prizes=None, max_iter: int = 1000):
    if problem in (1, 2):
        try:
            data = dataset_reader(instance_path)
            if problem == 1:
                if (
                    len(data["sources"]) != 1
                    or data["sources"].keys() != data["sinks"].keys()
                ):
                    raise ValueError("Min cut requires one source-sink pair.")
                A, b, c, y0 = build_min_cut(data)
            else:
                commodities = sorted(data["sources"])
                if prizes is None:
                    prizes = [1.0] * len(commodities)
                if len(prizes) != len(commodities) or not np.all(np.isfinite(prizes)):
                    raise ValueError("Provide one finite prize per commodity.")
                rewards = dict(zip(commodities, prizes))
                A, b, c, y0 = build_max_flow(data, rewards)

            start = perf_counter()
            x, y, z, iterations = primal_dual(A, b, c, y0, max_iter=max_iter)
            end = perf_counter()

            if not np.isfinite(z):
                raise ValueError("No convergence; increase --max-iter.")
            if problem == 2:
                z = -z
                y = -y

            print("=== Primal-Dual ===")
            print(f"Objective={z}, iterations={iterations}")
            print(f"Execution time: {(end - start):.6f}s\n")
            print(
                "Nonzero primal variables (column IDs):\n",
                {i: float(value) for i, value in enumerate(x) if abs(value) > 1e-9},
            )
            print(
                "Nonzero dual variables (row IDs):\n",
                {i: float(value) for i, value in enumerate(y) if abs(value) > 1e-9},
            )

            # PuLP.
            prob = build_problem(A, b, c)

            start = perf_counter()
            stats = prob.solve(pulp.COIN_CMD(msg=False))
            end = perf_counter()

            z = pulp.value(prob.objective)
            if problem == 2 and z is not None:
                z = -z

            print("\n\n=== PuLP ===")
            print(f"Objective={z}, status={stats.status_str}")
            print(f"Execution time: {(end - start):.6f}s\n")

            x_pulp = {}
            for v in prob.variables():
                j = int(v.name.split("_")[-1])
                if v.varValue is not None and abs(v.varValue) > 1e-9:
                    x_pulp[j] = float(v.varValue)

            print(
                "Nonzero primal variables (column IDs):\n", dict(sorted(x_pulp.items()))
            )

            cons = prob.constraints()
            cons = list(cons.values()) if isinstance(cons, dict) else list(cons)

            y_pulp = {}
            for i, con in enumerate(cons):
                if con.pi is not None and abs(con.pi) > 1e-9:
                    if problem == 1:
                        y_pulp[i] = float(con.pi)
                    else:
                        y_pulp[i] = -float(con.pi)

            print("Nonzero dual variables (row IDs):\n", dict(sorted(y_pulp.items())))

            # Stoer-Wagner - Min Cut.
            if problem == 1:
                weights = build_adjacency_matrix(data)

                start = perf_counter()
                cut, group = stoer_wagner(weights)
                end = perf_counter()

                print("\n\n=== Stoer-Wagner ===")
                print(f"Global cut={cut}")
                print(f"Execution time: {(end - start):.6f}s")
                print("Cut side (node IDs):", sorted(v + 1 for v in group))
                print("Global and s-t cuts may differ.")
        except (OSError, ValueError, KeyError, IndexError, np.linalg.LinAlgError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if problem == 0:
        # First example.
        print(
            "Solving: min 2x + 3y\nsubject to:\n\tx + y >= 5\n\t2x + y >= 7\n\tx, y >= 0\n"
        )

        A, b, c, y = example_1()
        try:
            start = perf_counter()
            x, y, z, iterations = primal_dual(A, b, c, y)
            end = perf_counter()
        except ValueError as e:
            print(e)
            sys.exit(1)

        print(f"Objective={z}, iterations={iterations}")
        print(f"Execution time: {(end - start):.6f}s\n")
        print(
            "Nonzero primal variables (column IDs):\n",
            {i: float(value) for i, value in enumerate(x) if abs(value) > 1e-9},
        )
        print(
            "Nonzero dual variables (row IDs):\n",
            {i: float(value) for i, value in enumerate(y) if abs(value) > 1e-9},
        )

        # Second example.
        print(
            "Solving: min x + 3y\nsubject to:\n\tx + 2y >= 2\n\t2x + y >= 2\n\tx, y >= 0\n"
        )

        A, b, c, y = example_2()
        try:
            start = perf_counter()
            x, y, z, iterations = primal_dual(A, b, c, y)
            end = perf_counter()
        except ValueError as e:
            print(e)
            sys.exit(1)

        print(f"Objective={z}, iterations={iterations}")
        print(f"Execution time: {(end - start):.6f}s\n")
        print(
            "Nonzero primal variables (column IDs):\n",
            {i: float(value) for i, value in enumerate(x) if abs(value) > 1e-9},
        )
        print(
            "Nonzero dual variables (row IDs):\n",
            {i: float(value) for i, value in enumerate(y) if abs(value) > 1e-9},
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args.problem, args.instance_path, args.prizes, args.max_iter)
