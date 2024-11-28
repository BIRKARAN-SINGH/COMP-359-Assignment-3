from scipy.optimize import linprog

c = [-3, -2]  
A = [[1, 2], [2, 1]]  
b = [10, 12] 
bounds = [(0, None), (0, None)] 

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='simplex')
print("Optimal Solution:", result.x)
print("Optimal Value:", -result.fun) 
