import numpy as np
import json
import os
from pathlib import Path

DATA_DIR = Path(os.path.join(os.getcwd(), "src", "SimulatedAnnealing", "data_extracted"))

class TSPData:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.nodes_path = self.data_dir / "tsp_nodes.json"
        self.distance_path = self.data_dir / "tsp_distance.npy"
        self.duration_path = self.data_dir / "tsp_duration.npy"
        self.steps_path = self.data_dir / "tsp_steps.npy"
        self._load_all()

    def _load_all(self):
        with open(self.nodes_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)
        self.city_names = nodes["names"]
        self.coords = nodes["coords"]
        self.distance = np.load(self.distance_path)
        self.duration = np.load(self.duration_path)
        self.steps_count = np.load(self.steps_path)
        self.name_to_idx = {name: i for i, name in enumerate(self.city_names)}

    def get_matrix(self, metric: str) -> np.ndarray:
        if metric == "distance":
            return self.distance
        elif metric == "duration":
            return self.duration
        elif metric == "steps_count":
            return self.steps_count
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_city_index(self, name: str) -> int:
        return self.name_to_idx[name]

    def get_city_coord(self, name: str):
        idx = self.name_to_idx[name]
        return self.coords[idx]

    def get_all_coords(self):
        return self.coords

    def get_all_names(self):
        return self.city_names

if __name__ == "__main__":
    tsp = TSPData()
    print("Cities:", tsp.get_all_names())
    print("Coords:", tsp.get_all_coords())
    print("Distance matrix shape:", tsp.distance.shape)
    print("Duration matrix shape:", tsp.duration.shape)
    print("Steps matrix shape:", tsp.steps_count.shape)
    # Example: get distance from Amstetten to Bregenz
    i = tsp.get_city_index("Amstetten")
    j = tsp.get_city_index("Bregenz")
    print(f"Distance Amstetten <-> Bregenz: {tsp.distance[i, j]}")
