import numpy as np
import matplotlib.pyplot as plt

def simplex(c, A, b):
    """
    Solves the linear programming problem:
    Maximize: c^T x
    Subject to: A x <= b, x >= 0
    using the Simplex algorithm.

    Parameters:
    c (numpy.ndarray): Coefficients of the objective function.
    A (numpy.ndarray): Coefficients of the inequality constraints.
    b (numpy.ndarray): Right-hand side values of the constraints.

    Returns:
    tuple: Optimal solution vector and the maximum value of the objective function.
    """
    # Number of constraints (m) and variables (n)
    m, n = A.shape

    # Construct the initial tableau
    tableau = np.hstack((A, np.eye(m), b.reshape(-1, 1)))
    c_extended = np.hstack((c, np.zeros(m + 1)))
    tableau = np.vstack((tableau, c_extended))

    # Iterative pivoting
    while True:
        # Step 1: Identify the entering variable (most negative coefficient in the objective row)
        entering = np.argmin(tableau[-1, :-1])
        if tableau[-1, entering] >= 0:
            # Optimal solution found
            break

        # Step 2: Identify the leaving variable using the minimum positive ratio test
        ratios = tableau[:-1, -1] / tableau[:-1, entering]
        valid_ratios = np.where(ratios > 0, ratios, np.inf)
        leaving = np.argmin(valid_ratios)
        if valid_ratios[leaving] == np.inf:
            raise ValueError("The linear program is unbounded.")

        # Step 3: Perform the pivot operation
        pivot = tableau[leaving, entering]
        tableau[leaving, :] /= pivot
        for i in range(m + 1):
            if i != leaving:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]

    # Extract the solution
    solution = np.zeros(n)
    for i in range(n):
        column = tableau[:-1, i]
        if np.count_nonzero(column) == 1 and np.any(column == 1):
            solution[i] = tableau[np.where(column == 1)[0][0], -1]

    # Optimal value of the objective function
    optimal_value = tableau[-1, -1]
    return solution, optimal_value

def plot_feasible_region(A, b):
    """
    Plots the feasible region defined by the constraints A x <= b.

    Parameters:
    A (numpy.ndarray): Coefficients of the inequality constraints.
    b (numpy.ndarray): Right-hand side values of the constraints.
    """
    x_vals = np.linspace(0, 10, 400)
    plt.figure(figsize=(8, 6))

    for i in range(A.shape[0]):
        if A[i, 1] != 0:
            y_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
            plt.plot(x_vals, y_vals, label=f'Constraint {i + 1}')
            plt.fill_between(x_vals, y_vals, 10, alpha=0.1)
        else:
            plt.axvline(x=b[i] / A[i, 0], label=f'Constraint {i + 1}')

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Feasible Region')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example problem:
    # Maximize Z = 3x1 + 2x2
    # Subject to:
    # x1 + x2 <= 4
    # 2x1 + x2 <= 5
    # x1, x2 >= 0

    c = np.array([3, 2])
    A = np.array([[1, 1], [2, 1]])
    b = np.array([4, 5])

    try:
        solution, max_value = simplex(c, A, b)
        print("Optimal solution:", solution)
        print("Maximum value of Z:", max_value)
    except ValueError as e:
        print(e)

    plot_feasible_region(A, b)
