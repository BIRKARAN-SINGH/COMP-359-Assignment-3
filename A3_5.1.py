from scipy.optimize import linprog
c = [-40, -50]

A = [[2, 3], [1, 2]]

b = [120, 80]

bounds = [(0, None), (0, None)]

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='simplex')

print("Optimal Solution (x, y):", result.x)
print("Optimal Profit (Z):", -result.fun)


