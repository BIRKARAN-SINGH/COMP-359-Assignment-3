import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 12, 400)
y1 = (10 - x) / 2  # Constraint 1: x + 2y <= 10
y2 = 12 - 2 * x    # Constraint 2: 2x + y <= 12


plt.plot(x, y1, label='x + 2y <= 10')
plt.plot(x, y2, label='2x + y <= 12')


y = np.minimum(y1, y2)
plt.fill_between(x, y, 0, where=(y >= 0), color='grey', alpha=0.5, label='Feasible Region')


plt.scatter(4.67, 2.67, color='red', label='Optimal Solution (4.67, 2.67)')

plt.xlim(0, 12)
plt.ylim(0, 6)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Graph of Constraints and Feasible Region')
plt.grid()
plt.show()
