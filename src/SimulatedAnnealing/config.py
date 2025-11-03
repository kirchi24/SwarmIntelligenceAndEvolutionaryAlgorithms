import os
from dynaconf import Dynaconf

# BASE_DIR ist Projekt-Root (eine Ebene über src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

settings = Dynaconf(
    settings_files=[
        os.path.join(BASE_DIR, "settings.toml"),
        os.path.join(BASE_DIR, ".secrets.toml"),
    ],
    environments=True,
    merge_enabled=True
)

# Absoluter Pfad zu den Daten
settings.coordinates_path = os.path.join(BASE_DIR, "src", "SimulatedAnnealing", "data", "city_coordinates.json")
settings.summary_path = os.path.join(BASE_DIR, "src", "SimulatedAnnealing", "data", "routes_summary.json")
settings.routes_path = os.path.join(BASE_DIR, "src", "SimulatedAnnealing", "data")
