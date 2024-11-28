from scipy.optimize import linprog

c = [3, 2]
A = [[-2, -1], [-1, -2]]
b = [-8, -10]

bounds = [(0, None), (0, None)]
result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='simplex')

print("Optimal Solution (x, y):", result.x)
print("Minimum Cost (Z):", result.fun)

