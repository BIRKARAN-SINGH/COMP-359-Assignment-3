import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the feasible region
x = np.linspace(0, 12, 400)
y1 = (10 - x) / 2
y2 = 12 - 2 * x
y = np.minimum(y1, y2)

# Corner points of the feasible region
points = [(0, 0), (0, 5), (4.67, 2.67), (6, 0)]

# Create a plot
fig, ax = plt.subplots()
plt.plot(x, y1, label='x + 2y <= 10')
plt.plot(x, y2, label='2x + y <= 12')
plt.fill_between(x, y, 0, where=(y >= 0), color='grey', alpha=0.5)
plt.xlim(0, 12)
plt.ylim(0, 6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simplex Algorithm Visualization')
plt.grid()
sc = ax.scatter([], [], color='red', s=100)

# Animation function
def update(frame):
    sc.set_offsets(points[:frame+1])

ani = FuncAnimation(fig, update, frames=len(points), interval=1000, repeat=False)
plt.legend()
plt.show()
