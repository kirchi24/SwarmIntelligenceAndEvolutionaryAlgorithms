import random
import math
from pathlib import Path
from .load_tsp_data import TSPData
import json

# --- Pfad zu den GeoJSON-Routen ---
ROUTES_DIR = Path("src/SimulatedAnnealing/data")

# --- TSP-Daten laden ---
tsp = TSPData()

# --- Helferfunktionen für SA ---
def total_distance(route, dist_matrix):
    distance = 0
    for i in range(len(route)):
        distance += dist_matrix[route[i]][route[(i + 1) % len(route)]]
    return distance

def get_neighbor(route):
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

# --- Route aus GeoJSON laden ---
def get_route_coords(start_city, end_city):
    """
    Liefert eine Liste von [lon, lat]-Koordinaten zwischen zwei Städten
    anhand der gespeicherten GeoJSON-Routen.
    """
    # Key aus Start und Ziel
    route_key = f"{start_city.lower()}_{end_city.lower()}"
    reverse_key = f"{end_city.lower()}_{start_city.lower()}"

    # Suche nach der Datei
    for file_path in ROUTES_DIR.glob("*.json"):
        if file_path.stem == route_key or file_path.stem == reverse_key:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", [])
            if not features:
                return []
            coords = features[0].get("geometry", {}).get("coordinates", [])
            if file_path.stem == reverse_key:
                coords = list(reversed(coords))
            return coords

    # Fallback: gerade Linie zwischen den Städten
    return [tsp.get_city_coord(start_city), tsp.get_city_coord(end_city)]
