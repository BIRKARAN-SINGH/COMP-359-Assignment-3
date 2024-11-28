import numpy as np

def simplex(c, A, b):
    # Negate the coefficients of the objective function to convert maximization to minimization
    tableau = np.hstack([A, np.eye(len(A)), b.reshape(-1, 1)])
    tableau = np.vstack([tableau, np.hstack([-c, np.zeros(len(b) + 1)])])

    while np.any(tableau[-1, :-1] < 0):  # While there are negative entries in the objective row
        pivot_col = np.argmin(tableau[-1, :-1])  # Most negative column
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]  # Calculate ratios
        pivot_row = np.where(ratios == np.min(ratios[ratios > 0]))[0][0]  # Choose pivot row

        # Normalize pivot row
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        
        # Update other rows
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

    solution = tableau[:-1, -1]  # Solution values for decision variables
    return solution, tableau[-1, -1]  # Return solution and optimal value

# Example problem: Maximize Z = 3x + 2y subject to constraints
c = np.array([3, 2])  # Coefficients of the objective function
A = np.array([[1, 2], [2, 1]])  # Coefficients of the constraints
b = np.array([10, 12])  # RHS of the constraints

solution, optimal_value = simplex(c, A, b)
print("Optimal Solution:", solution)
print("Optimal Value:", optimal_value)
