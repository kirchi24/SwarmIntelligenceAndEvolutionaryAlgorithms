import numpy as np
import json
import os
from pathlib import Path

def load_cities() -> dict:
    path = os.path.join(
        os.getcwd(), "src", "SimulatedAnnealing", "data", "city_coordinates.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_routes() -> dict:
    data_dir = Path(os.path.join(
        os.getcwd(), "src", "SimulatedAnnealing", "data"
    ))
    json_files = [
        f for f in data_dir.glob("*.json") if f.name not in ["city_coordinates.json", "routes_summary.json"]
    ]
    routes = {}
    for file_path in json_files:
        route_name = file_path.stem
        segments = extract(file_path.name)
        if segments:
            routes[route_name] = segments
    return routes

def extract(file_name: str):
    path = os.path.join(
        os.getcwd(), "src", "SimulatedAnnealing", "data", file_name
    )
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    coords = j.get("metadata", {}).get("query", {}).get("coordinates", [])
    start_coord = coords[0] if len(coords) > 0 else None
    end_coord = coords[1] if len(coords) > 1 else None
    features = j.get("features", [])
    if not features:
        print(f"No features found in {file_name}")
        return []
    out = []
    for feature in features:
        props = feature.get("properties", {})
        segments = props.get("segments", [])
        for seg in segments:
            distance = seg.get("distance")
            duration = seg.get("duration")
            steps_count = len(seg.get("steps", []))
            out.append({
                "distance": distance,
                "duration": duration,
                "steps_count": steps_count,
                "start_coord": start_coord,
                "end_coord": end_coord
            })
    return out

def build_tsp_data():
    cities = load_cities()  # {name: [lon, lat]}
    city_names = sorted(cities.keys())
    name_to_idx = {name: i for i, name in enumerate(city_names)}
    coords_list = [cities[name] for name in city_names]
    N = len(city_names)

    # Matrizen initialisieren
    distance_matrix = np.full((N, N), np.inf)
    duration_matrix = np.full((N, N), np.inf)
    steps_matrix = np.full((N, N), np.inf)

    # Routen laden
    routes = load_routes()  # {route_name: [segment_dicts]}
    for route_segments in routes.values():
        for seg in route_segments:
            start = seg["start_coord"]
            end = seg["end_coord"]
            # Finde Namen zu Koordinaten
            start_name = next((n for n, c in cities.items() if c == start), None)
            end_name = next((n for n, c in cities.items() if c == end), None)
            if start_name is None or end_name is None:
                print(f"Warn: Could not match coords {start} or {end} to city name.")
                continue
            i, j = name_to_idx[start_name], name_to_idx[end_name]
            d = seg["distance"]
            t = seg["duration"]
            s = seg["steps_count"]
            # Symmetrisch eintragen
            distance_matrix[i, j] = distance_matrix[j, i] = d
            duration_matrix[i, j] = duration_matrix[j, i] = t
            steps_matrix[i, j] = steps_matrix[j, i] = s
    # Speichern
    out_dir = Path(os.path.join(os.getcwd(), "src", "SimulatedAnnealing", "data_extracted"))
    with open(out_dir / "tsp_nodes.json", "w", encoding="utf-8") as f:
        json.dump({"names": city_names, "coords": coords_list}, f, indent=2)
    np.save(out_dir / "tsp_distance.npy", distance_matrix)
    np.save(out_dir / "tsp_duration.npy", duration_matrix)
    np.save(out_dir / "tsp_steps.npy", steps_matrix)
    print(f"Saved nodes to tsp_nodes.json and matrices to tsp_distance.npy, tsp_duration.npy, tsp_steps.npy")

def main():
    build_tsp_data()

if __name__ == "__main__":
    main()
