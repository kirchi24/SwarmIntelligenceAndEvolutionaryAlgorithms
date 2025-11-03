import os
from dynaconf import Dynaconf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

settings = Dynaconf(
    settings_files=[
        os.path.join(BASE_DIR, "settings.toml"),
        os.path.join(BASE_DIR, ".secrets.toml"),
    ],
    environments=True,
    merge_enabled=True
)