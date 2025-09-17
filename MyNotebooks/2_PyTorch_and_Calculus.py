import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt


def f(x):
    return 3 * x**2 - 4 * x


for h in 10.0 ** np.arange(-1, -6, -1):
    print(f"h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}")

# Use the svg format to display a plot in Jupyter.
# backend_inline.set_matplotlib_formats('svg')

# Set the figure size for matplotlib.
plt.rcParams["figure.figsize"] = (3.5, 2.5)

# Set the axes for matplotlib.
axes = plt.gca()
axes.cla()
axes.set_xlabel("x")
axes.set_ylabel("f(x)")

axes.set_xscale("linear")
axes.set_yscale("linear")

axes.set_xlim(0, 3)
axes.set_ylim(-5, 15)

# Plot the function.
x = np.arange(0, 3, 0.1)
axes.plot(x, f(x))
axes.plot(x, 2 * x - 3, color="purple", linestyle="--")
legend = ["f(x)", "Tangent line (x=1)"]
axes.legend(legend)
axes.grid()


# Define function
def f(x, y):
    return x**2 * y + np.sin(x * y)


# Partial derivative w.r.t x (finite difference)
def partial_x(x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


# Partial derivative w.r.t y
def partial_y(x, y, h=1e-5):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


print("∂f/∂x =", partial_x(2.0, 3.0))
print("∂f/∂y =", partial_y(2.0, 3.0))


# Create grid
x_vals = np.linspace(-5, 5, 15)
y_vals = np.linspace(-5, 5, 15)
X, Y = np.meshgrid(x_vals, y_vals)

# Function
Z = X**2 + Y**2

# Gradients
grad_x = 2 * X
grad_y = 2 * Y

# 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot surface
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)

# Plot arrows (gradient vectors) with arrowheads
q = ax.quiver(
    X,
    Y,
    Z,
    grad_x,
    grad_y,
    np.zeros_like(Z),
    length=0.5,
    arrow_length_ratio=0.3,
    color="red",
    normalize=True,
)

# Labels
ax.set_title("3D Gradient Field ")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
plt.show()
