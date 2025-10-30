from pathlib import Path
import json
import sys
import os

def load_cities() -> dict:
    path = os.path.join(
        os.getcwd(), "src", "SimulatedAnnealing", "data", "city_coordinates.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_routes() -> dict:
    # get all json files in data directory
    data_dir = Path(os.path.join(
        os.getcwd(), "src", "SimulatedAnnealing", "data"
    ))
    
    # filter json files excluding city_coordinates.json
    json_files = [
        f for f in data_dir.glob("*.json") if f.name != "city_coordinates.json"
    ]

    # extreact data from each file
    routes = {}
    for file_path in json_files:
        route_name = file_path.stem  # filename without extension
        routes[route_name] = extract(file_path.name)    

    return routes

def extract(file_name: str):
    path = os.path.join(
        os.getcwd(), "src", "SimulatedAnnealing", "data", file_name
    ) 
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    # metadata query coordinates (start, end)
    coords = j.get("metadata", {}).get("query", {}).get("coordinates", [])
    start_coord = coords[0] if len(coords) > 0 else None
    end_coord = coords[1] if len(coords) > 1 else None

    features = j.get("features", [])
    if not features:
        print("No features found")
        return

    # For each feature, list segments with distance, duration and number of steps
    out = []
    for fi, feature in enumerate(features):
        props = feature.get("properties", {})
        segments = props.get("segments", [])
        for si, seg in enumerate(segments):
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

def main():
    try:
        cities = load_cities()
        routes = load_routes()
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    print(f"Loaded {len(cities)} cities\n")
    for name in sorted(cities):
        coords = cities[name]
        # coords expected as [longitude, latitude]
        print(f"{name}: lon={coords[0]}, lat={coords[1]}")
    
    print("\nLoaded routes:")
    for name, route in sorted(routes.items()):
        print(f"{name}: {route} segments")

if __name__ == "__main__":
    main()
