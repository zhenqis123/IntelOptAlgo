# GA_TSP.py
# Genetic Algorithm for TSP
# Author: Ziheng Xi
# Date: 2024-12-18

import numpy as np
import matplotlib.pyplot as plt
import random

# check if the path is valid
def is_valid_path(path):
    if not isinstance(path, list) or len(path) == 0:
        return False
    max_index = max(path)
    min_index = min(path)
    if min_index != 0 or max_index != len(path) - 1:
        return False
    return set(path) == set(range(len(path)))
# generate cities
def generate_cities(num_cities=50):
    cities = np.random.rand(num_cities, 2) * 100  # generate random coordinates in the range of 0-100
    return cities

# calculate distance
def calculate_distance(cities, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += np.linalg.norm(cities[path[i]] - cities[path[i+1]])
    # back to start
    distance += np.linalg.norm(cities[path[-1]] - cities[path[0]])
    return distance

# genetic algorithm
def genetic_algorithm(cities, pop_size=100, generations=500, mutation_rate=0.02):
    num_cities = len(cities)

    # initialize population: random generate path
    population = [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

    def selection(population, fitness):
        """ roulette wheel selection """
        total_fitness = sum(fitness)
        probs = [f / total_fitness for f in fitness]
        return population[np.random.choice(range(len(population)), p=probs)]

    def pmx_crossover(parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end+1] = parent1[start:end+1]

        for i in range(start, end+1):
            if parent2[i] not in child:
                j = i
                while start <= j <= end:
                    j = parent2.index(parent1[j])
                child[j] = parent2[i]

        for i in range(size):
            if child[i] == -1:
                child[i] = parent2[i]

        return child

    def mutate(path):
        """ two-point exchange mutation """
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]
        return path

    # genetic algorithm main loop
    best_path = None
    best_distance = float('inf')
    fitness_history = []
    average_distances = []

    for gen in range(generations):
        # calculate fitness
        distances = [calculate_distance(cities, path) for path in population]
        fitness = [1 / d for d in distances]

        # record current best solution
        current_best_idx = np.argmin(distances)
        current_best_distance = distances[current_best_idx]
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_path = population[current_best_idx]

        # calculate average distance
        average_distance = np.mean(distances)
        average_distances.append(average_distance)

        # generate next generation
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = selection(population, fitness)
            parent2 = selection(population, fitness)
            child1 = pmx_crossover(parent1, parent2)
            child2 = pmx_crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population
        fitness_history.append(best_distance)

        # output current best and average solution every 10 generations
        if gen % 10 == 0:
            print(f"Generation {gen}, Best Distance: {best_distance:.2f}, Average Distance: {average_distance:.2f}")

    return best_path, best_distance, fitness_history, average_distances

# visualize best path and performance
def plot_results(cities, best_path, fitness_history, average_distances):
    # plot path
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Best Path")
    path = np.array([cities[i] for i in best_path] + [cities[best_path[0]]])
    
    # Plot the path with arrows
    for i in range(len(path) - 1):
        plt.quiver(path[i, 0], path[i, 1], path[i+1, 0] - path[i, 0], path[i+1, 1] - path[i, 1],
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.01)
    
    # Mark start and end points
    plt.scatter(path[0, 0], path[0, 1], color='green', label='Start', zorder=5)
    plt.scatter(path[-1, 0], path[-1, 1], color='red', label='End', zorder=5)
    
    plt.legend()

    # plot fitness change
    plt.subplot(1, 2, 2)
    plt.title("Fitness Over Generations")
    plt.plot(fitness_history, label='Best Distance')
    # plt.plot(average_distances, label='Average Distance', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig('GA_TSP.png')

def run_experiments(cities, num_experiments=10, pop_size=100, generations=600, mutation_rate=0.02):
    all_best_distances = []
    all_average_distances = []

    for _ in range(num_experiments):
        best_path, best_distance, fitness_history, average_distances = genetic_algorithm(
            cities, pop_size, generations, mutation_rate
        )
        all_best_distances.append(best_distance)
        all_average_distances.append(average_distances)

    return all_best_distances, all_average_distances

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    num_cities = 50
    cities = generate_cities(num_cities)

    num_experiments = 20
    all_best_distances, all_average_distances = run_experiments(cities, num_experiments)

    best_perf = 10000000
    worst_perf = -1
    best_perf_idx = 0
    worst_perf_idx = 0
    for i, best_distance in enumerate(all_best_distances):
        if best_distance < best_perf:
            best_perf = best_distance
            best_perf_idx = i
        if best_distance > worst_perf:
            worst_perf = best_distance
            worst_perf_idx = i
    average_perf = np.mean(all_best_distances)
    var_perf = np.var(all_best_distances)
    print(f"Best performance: {best_perf:.2f} (Experiment {best_perf_idx+1})")
    print(f"Worst performance: {worst_perf:.2f} (Experiment {worst_perf_idx+1})")
    print(f"Average performance: {average_perf:.2f}")
    print(f"Variance of performance: {var_perf:.2f}")

    # Optionally, plot results of the last experiment
    best_path, _, fitness_history, average_distances = genetic_algorithm(cities)
    plot_results(cities, best_path, fitness_history, average_distances)

