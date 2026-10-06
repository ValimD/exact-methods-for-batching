import argparse
import sys

from methods.primal_dual import primal_dual
from problems.example import example_1, example_2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Run the primal-dual algorithm for predefined problems.")
    
    parser.add_argument(
        "problem",
        type=int,
        choices=[0, 1, 2],
        help="Problem to solve: 0 (Examples), 1 (Min Cut), 2 (Max Flow)."
    )
    
    parser.add_argument(
        "instance_path",
        nargs="?",
        help="Path to the problem instance."
    )
    
    args = parser.parse_args()

    if args.problem != 0 and args.instance_path is None:
        parser.error("instance_path is required for problems 1 and 2.")

    return args

def main(problem: int, instance_path: str | None):
    if problem == 0:
        # First example.
        print("Solving: min 2x + 3y\nsubject to:\n\tx + y >= 5\n\t2x + y >= 7\n\tx, y >= 0\n")

        A, b, c, y = example_1()
        try:
            x, y, z, iter = primal_dual(A, b, c, y)
        except ValueError as e:
            print(e)
            sys.exit(1)

        print(f"Problem solved in {iter} iterations.")
        print(f"Objective function: {z}.")
        print(f"Primal variables: {x}, dual variables: {y}.\n\n")

        # Second example.
        print("Solving: min x + 3y\nsubject to:\n\tx + 2y >= 2\n\t2x + y >= 2\n\tx, y >= 0\n")

        A, b, c, y = example_2()
        try:
            x, y, z, iter = primal_dual(A, b, c, y)
        except ValueError as e:
            print(e)
            sys.exit(1)
        
        print(f"Problem solved in {iter} iterations.")
        print(f"Objective function: {z}.")
        print(f"Primal variables: {x}, dual variables: {y}.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.problem, args.instance_path)