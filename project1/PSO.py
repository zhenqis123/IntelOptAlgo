# PSO.py
# PSO algorithm for optimization
# objective function: Schwefel function
# Author: Ziheng Xi
# Date: 2024-12-18

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ackley function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    if isinstance(x, list):
        x = np.array(x)
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

def schwefel(x):
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
# Visualize the Ackley function
def visualize(bounds, function):
    x = np.linspace(bounds[0, 0], bounds[0, 1], 200)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[function([xi, yi]) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.set_title("Ackley Function")
    # ax.set_title("Schwefel Function")
    ax.set_title(function.__name__)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(X, Y)")
    # plt.show()
    plt.savefig(f'project1/{function.__name__}.png')

# Hybrid PSO-DE algorithm
def pso(objective, bounds, num_particles=30, max_iter=500, w=0.9, c1=2, c2=2):
    dim = len(bounds)
    # Initialize particles
    particles = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(num_particles, dim))
    velocities = np.random.uniform(-1, 1, size=(num_particles, dim))
    personal_best_positions = np.copy(particles)
    personal_best_scores = np.array([objective(p) for p in particles])

    # Initialize global best
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = np.copy(personal_best_positions[global_best_index])
    global_best_score = personal_best_scores[global_best_index]

    # Store convergence curve
    convergence_curve = []

    for t in range(max_iter):
        w_t = w - t * (w - 0.4) / max_iter
        for i in range(num_particles):
            # PSO update
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w_t * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - particles[i]) +
                             c2 * r2 * (global_best_position - particles[i]))
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
            fitness = objective(particles[i])

            # Update personal and global best
            if fitness < personal_best_scores[i]:
                personal_best_positions[i] = np.copy(particles[i])
                personal_best_scores[i] = fitness
            if fitness < global_best_score:
                global_best_position = np.copy(particles[i])
                global_best_score = fitness

        convergence_curve.append(global_best_score)

    return global_best_score, convergence_curve, global_best_position

# Experiment parameters
bounds_schwefel = np.array([[-500, 500], [-500, 500]])
bounds_ackley = np.array([[-32.768, 32.768], [-32.768, 32.768]])

num_experiments = 20
max_iter = 200

# Visualize the functions
visualize(bounds_schwefel, schwefel)
visualize(bounds_ackley, ackley)

# Update Schwefel bounds to 10 dimensions
bounds_schwefel = np.array([[-500, 500]] * 5)  # 10-dimensional search space
bounds_ackley = np.array([[-32.768, 32.768]] * 5)  # 10-dimensional search space
# Run PSO for 20 experiments
results = []
all_curves = []
all_solutions = []

for exp in range(num_experiments):
    best_score, convergence_curve, best_solution = pso(ackley, bounds_ackley, max_iter=max_iter)
    results.append(best_score)
    all_curves.append(convergence_curve)
    all_solutions.append(best_solution)

# Statistical Results
mean_result = np.mean(results)
best_result = np.min(results)
worst_result = np.max(results)
variance_result = np.var(results)

print("Statistical Results after 20 Runs:")
print(f"Mean Performance: {mean_result}")
print(f"Best Performance: {best_result}")
print(f"Worst Performance: {worst_result}")
print(f"Variance: {variance_result}")
print(f"Best Solution: {all_solutions[0]}")

# Plot typical convergence curve
plt.figure(figsize=(8, 5))
plt.plot(all_curves[0], label='Typical Convergence Curve')
plt.title('Convergence of PSO on Ackley Function')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness Value')
plt.legend()
plt.grid()
plt.savefig('project1/PSO_Ackley.png')