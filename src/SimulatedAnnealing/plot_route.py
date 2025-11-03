import json
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from src.SimulatedAnnealing.config import settings

def fetch_route(city_from: str, city_to: str):
    file_name_alt1 = os.path.join(settings.routes_path, f"route_{city_from}_{city_to}.json")
    file_name_alt2 = os.path.join(settings.routes_path, f"route_{city_to}_{city_from}.json")

    if os.path.exists(file_name_alt1):
        route_file = file_name_alt1
    elif os.path.exists(file_name_alt2):
        route_file = file_name_alt2
    else:
        raise FileNotFoundError(f"No route file found for {city_from} ↔ {city_to}")

    with open(route_file, "r") as f:
        route_data = json.load(f)

    coords = route_data["features"][0]["geometry"]["coordinates"]
    route_line = LineString(coords)
    route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
    return route_gdf


def plot_route(route: list[str]):
    with open(settings.summary_path, "r", encoding="utf-8") as f:
        routes = json.load(f)

    with open(settings.coordinates_path, "r", encoding="utf-8") as f:
        coords = json.load(f)

    world = gpd.read_file(settings.gpd_url)
    austria = world[world["NAME"] == "Austria"]

    _, ax = plt.subplots(figsize=(8, 6))
    austria.plot(ax=ax, color="beige", edgecolor="gray")
    cmap = plt.get_cmap("viridis", len(route)-1)

    for i in range(len(route)-1):
        color = cmap(i)
        current_route = fetch_route(route[i], route[i+1])
        current_route.plot(ax=ax, color=color, linewidth=2)

    for city, (lon, lat) in coords.items():
        ax.scatter(lon, lat, color="black", s=50)
        ax.text(lon + 0.1, lat + 0.1, city, fontsize=8)

    ax.set_title("TSP Route Across Austrian Cities", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()
