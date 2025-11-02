import random
import math
from pathlib import Path
from .load_tsp_data import TSPData
import json

ROUTES_DIR = Path("src/SimulatedAnnealing/data")
tsp = TSPData()

def total_distance(route, dist_matrix):
    return sum(dist_matrix[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))


def nearest_neighbor_solution(dist_matrix, start=0):
    """Erstellt eine gierige Startlösung."""
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
    new_route = route.copy()
    i, j = sorted(random.sample(range(len(route)), 2))
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route


def reinsertion(route):
    """Nimmt eine Stadt und fügt sie an anderer Stelle wieder ein."""
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    city = new_route.pop(i)
    new_route.insert(j, city)
    return new_route


def get_neighbor(route):
    """Mischung aus 2-opt, Swap, und Reinsertion für bessere Exploration."""
    r = random.random()
    if r < 0.4:
        return two_opt(route)
    elif r < 0.8:
        return reinsertion(route)
    else:
        # swap fallback
        i, j = random.sample(range(len(route)), 2)
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route


# --- Hauptalgorithmus ---
def simulated_annealing(
    dist_matrix,
    T_start=2000,
    T_end=1,
    alpha=0.999,
    max_iter=50000,
    reheating_factor=1.1,
    stagnation_limit=2000,
    return_history=False
):
    """
    Verbesserte Version von Simulated Annealing für TSP.
    - Startet mit einer gierigen Lösung
    - Adaptive Temperatursteuerung + Reheating
    """
    n = len(dist_matrix)
    current_route = nearest_neighbor_solution(dist_matrix)
    current_distance = total_distance(current_route, dist_matrix)

    best_route = current_route.copy()
    best_distance = current_distance
    best_history = [best_distance]

    T = T_start
    no_improve = 0

    for step in range(max_iter):
        new_route = get_neighbor(current_route)
        new_distance = total_distance(new_route, dist_matrix)
        delta = new_distance - current_distance

        # Akzeptanzregel mit adaptiver Wahrscheinlichkeit
        if delta < 0 or random.random() < math.exp(-delta / (T + 1e-9)):
            current_route = new_route
            current_distance = new_distance

            if current_distance < best_distance:
                best_route = current_route.copy()
                best_distance = current_distance
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        best_history.append(best_distance)

        # Temperaturupdate
        T *= alpha

        # Reheating falls Stagnation
        if no_improve > stagnation_limit:
            T *= reheating_factor
            no_improve = 0

        if T < T_end:
            break

    if return_history:
        return best_route, best_distance, best_history
    return best_route, best_distance


def get_sa_route_coords(best_route, tsp):
    """
    Wandelt eine SA-Stadtindex-Reihenfolge in Koordinaten um,
    indem die echten GeoJSON-Routen für jede Stadt-zu-Stadt-Verbindung genutzt werden.
    """
    coords = []
    for i in range(len(best_route)-1):
        start_idx = best_route[i]
        end_idx = best_route[i+1]
        start_city = tsp.city_names[start_idx]
        end_city = tsp.city_names[end_idx]

        # GeoJSON-Routen laden
        route_coords = get_route_coords(start_city, end_city)
        if not route_coords:
            # Fallback: gerade Linie
            route_coords = [tsp.get_city_coord(start_city), tsp.get_city_coord(end_city)]

        if coords:
            # Verbinde nahtlos: letztes Ende der bisherigen Route ist Start der nächsten
            coords.extend(route_coords[1:])
        else:
            coords.extend(route_coords)
    return coords


def get_route_coords(start_city, end_city):
    """Liefert eine Liste von [lon, lat]-Koordinaten zwischen zwei Städten."""
    start_city_lower = start_city.lower()
    end_city_lower = end_city.lower()

    # Suche nach allen passenden Dateien
    for file_path in ROUTES_DIR.glob("route_*.json"):
        fname = file_path.stem.lower()  # z.B. route_traun_graz
        # Prüfen, ob sowohl Start als auch Ziel im Dateinamen vorkommen
        if start_city_lower in fname and end_city_lower in fname:
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

    # Fallback: gerade Linie
    return [tsp.get_city_coord(start_city), tsp.get_city_coord(end_city)]

def get_all_routes(best_route, tsp, start_city=None, end_city=None):
    """
    Gibt Koordinaten für:
      - SA-Route: anhand der echten Straßen
      - Echte Route zwischen Start- und Zielstadt (optional)
    """
    sa_coords = get_sa_route_coords(best_route, tsp)
    real_coords = []
    if start_city and end_city:
        real_coords = get_route_coords(start_city, end_city)
    return sa_coords, real_coords