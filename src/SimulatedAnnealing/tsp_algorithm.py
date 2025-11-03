import random
import math
from pathlib import Path
import numpy as np
from .load_tsp_data import TSPData
import json

ROUTES_DIR = Path("src/SimulatedAnnealing/data")
tsp = TSPData()


def total_distance(route, dist_matrix):
    """Summe der Streckenlängen. Gibt inf zurück, falls ungültige Werte vorkommen."""
    distance = 0.0
    n = len(route)
    if n < 2:
        return float("inf")
    for i in range(n - 1):
        d = dist_matrix[route[i]][route[i + 1]]
        if np.isnan(d) or np.isinf(d):
            return float("inf")
        distance += d
    # Rückkehr zur Startstadt
    d = dist_matrix[route[-1]][route[0]]
    if np.isnan(d) or np.isinf(d):
        return float("inf")
    distance += d
    return distance


def nearest_neighbor_solution(dist_matrix, start=0):
    """Gierige Startlösung ab Startstadt, Rundreise zurück."""
    n = len(dist_matrix)
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    while unvisited:
        last = route[-1]
        next_city = min(unvisited, key=lambda x: dist_matrix[last][x])
        route.append(next_city)
        unvisited.remove(next_city)
    return route


def two_opt(route):
    """Invertiert ein Segment der Route, Startstadt bleibt vorne."""
    new_route = route.copy()
    i, j = sorted(random.sample(range(1, len(route)), 2))
    new_route[i : j + 1] = reversed(new_route[i : j + 1])
    return new_route


def reinsertion(route):
    """Eine Stadt neu einfügen, Startstadt bleibt vorne."""
    new_route = route.copy()
    i, j = random.sample(range(1, len(route)), 2)
    city = new_route.pop(i)
    new_route.insert(j, city)
    return new_route


def swap(route):
    """Tauscht zwei Städte, Startstadt bleibt vorne."""
    new_route = route.copy()
    i, j = random.sample(range(1, len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def get_neighbor(route):
    """Wählt zufällig 2-opt, Reinsertion oder Swap für Nachbarschaft."""
    r = random.random()
    if r < 0.4:
        return two_opt(route)
    elif r < 0.8:
        return reinsertion(route)
    else:
        return swap(route)


def simulated_annealing(
    dist_matrix,
    start_city_index=0,
    T_start=3000,
    T_end=0.1,
    alpha=0.9995,
    max_iter=100000,
    reheating_factor=1.2,
    stagnation_limit=2500,
    neighborhood_boost=True,
    return_history=False,
):
    """
    Simulated Annealing für TSP:
    - Start bei fixierter Startstadt, Rundreise.
    - Dynamisches Reheating bei Stagnation.
    """
    n = len(dist_matrix)
    current_route = nearest_neighbor_solution(dist_matrix, start=start_city_index)
    current_distance = total_distance(current_route, dist_matrix)

    best_route = current_route.copy()
    best_distance = current_distance
    history = [best_distance]

    T = T_start
    no_improve = 0

    for _ in range(max_iter):
        if neighborhood_boost and no_improve > stagnation_limit // 4:
            neighbor = two_opt(current_route)
        else:
            neighbor = get_neighbor(current_route)

        new_distance = total_distance(neighbor, dist_matrix)
        delta = new_distance - current_distance

        if delta < 0 or random.random() < math.exp(-delta / (T + 1e-9)):
            current_route = neighbor
            current_distance = new_distance
            if current_distance < best_distance:
                best_route = current_route.copy()
                best_distance = current_distance
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        history.append(best_distance)
        T *= alpha

        if no_improve > stagnation_limit:
            T *= reheating_factor
            no_improve = 0

        if T < T_end:
            break

    if return_history:
        return best_route, best_distance, history
    return best_route, best_distance


def get_sa_route_coords(best_route, tsp):
    coords = []
    for i in range(len(best_route) - 1):
        start_idx = best_route[i]
        end_idx = best_route[i + 1]
        start_city = tsp.city_names[start_idx]
        end_city = tsp.city_names[end_idx]

        route_coords = get_route_coords(start_city, end_city)

        if not route_coords or len(route_coords) < 2:
            route_coords = [
                tsp.get_city_coord(start_city),
                tsp.get_city_coord(end_city),
            ]

        if coords:
            coords.extend(route_coords[1:])
        else:
            coords.extend(route_coords)

    start_city_idx = best_route[0]
    coords.append(tsp.get_city_coord(tsp.city_names[start_city_idx]))
    return coords


# Hilfsfunktion: prüft, ob eine Datei für die gegebene Richtung existiert
def find_coords(s_city, e_city):
    s_city_lower = s_city.lower()
    e_city_lower = e_city.lower()
    for file_path in ROUTES_DIR.glob("route_*.json"):
        fname = file_path.stem.lower()
        if s_city_lower in fname and e_city_lower in fname:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", [])
            if not features:
                return []
            coords = []
            for feature in features:
                geom_coords = feature.get("geometry", {}).get("coordinates", [])
                coords.extend(geom_coords)
            return coords
    return None


def get_route_coords(start_city, end_city):
    """Liefert Koordinaten zwischen zwei Städten."""

    coords = find_coords(start_city, end_city)
    if coords:
        return coords

    coords = find_coords(end_city, start_city)
    if coords:
        return coords[::-1]  # Route umdrehen, damit die Richtung stimmt

    # 3. Fallback: direkte Stadtkoordinaten
    return [tsp.get_city_coord(start_city), tsp.get_city_coord(end_city)]
