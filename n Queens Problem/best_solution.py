from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp


def main(board_size):
    solver = pywrapcp.Solver("n-queens")
    queens = [solver.IntVar(0, board_size - 1, "x%i" % i) for i in range(board_size)]

    solver.Add(solver.AllDifferent(queens))
    solver.Add(solver.AllDifferent([queens[i] + i for i in range(board_size)]))
    solver.Add(solver.AllDifferent([queens[i] - i for i in range(board_size)]))

    db = solver.Phase(queens, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
    solver.NewSearch(db)

    while solver.NextSolution():
        for i in range(board_size):
            for j in range(board_size):
                if queens[j].Value() == i:
                    print("Q", end=" ")
                else:
                    print("_", end=" ")
            print()
        print()
        break

    solver.EndSearch()

    print()
    print("Time:", solver.WallTime(), "ms")


initial_board_size = 24

if __name__ == "__main__":
    if len(sys.argv) > 1:
        initial_board_size = int(sys.argv[1])
    main(initial_board_size)
