from load_tsp_data import TSPData
import numpy as np
import random
import math


# Wähle drei Städtenamen
selected_cities = ["Amstetten", "Bregenz", "Graz"]

# Lade TSP-Daten
tsp = TSPData()

# Hole die Indizes der ausgewählten Städte
indices = [tsp.get_city_index(name) for name in selected_cities]

# Extrahiere die 3x3-Submatrix
dist_matrix = tsp.distance[np.ix_(indices, indices)]

def total_distance(route, dist_matrix):
    """Berechnet die Gesamtdistanz einer Route."""
    distance = 0
    for i in range(len(route)):
        distance += dist_matrix[route[i]][route[(i + 1) % len(route)]]
    return distance

def get_neighbor(route):
    """Erzeugt eine neue Route durch Vertauschen zweier Städte."""
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def simulated_annealing(dist_matrix, T_start=1000, T_end=1, alpha=0.995, max_iter=10000):
    n = len(dist_matrix)
    current_route = list(range(n))
    random.shuffle(current_route)
    current_distance = total_distance(current_route, dist_matrix)
    
    best_route = current_route.copy()
    best_distance = current_distance
    T = T_start

    for step in range(max_iter):
        new_route = get_neighbor(current_route)
        new_distance = total_distance(new_route, dist_matrix)
        delta = new_distance - current_distance

        # Akzeptanzregel
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_route = new_route
            current_distance = new_distance

            if current_distance < best_distance:
                best_route = current_route.copy()
                best_distance = current_distance

        T *= alpha
        if T < T_end:
            break

    return best_route, best_distance

# Aufruf
best_route, best_distance = simulated_annealing(dist_matrix)
print("Beste Route:", best_route)
print("Distanz:", best_distance)
