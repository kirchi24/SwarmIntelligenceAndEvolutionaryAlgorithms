# Swarm Intelligence and Evolutionary Algorithms

This repository is developed as part of the Swarm Intelligence and Evolutionary Algorithms course in the Data Science and Artificial Intelligence program at FH Joanneum Graz.

It contains various tasks and experiments implementing swarm intelligence and evolutionary algorithms, which are documented and interactively demonstrated using Streamlit. The goal is to make the functionality and behavior of these algorithms understandable and visually accessible.

## Project structure

The repository follows a simple layout to separate Streamlit pages (user-facing exercises/demos) from reusable source code and other project files. Current top-level files and folders:

- `streamlit_app.py` - Streamlit app entry point and app configuration.
- `pages/` - Streamlit multi-page files. Each page corresponds to an exercise or demo and is loaded by Streamlit when the app runs. Example: `pages/1_HillClimbingAlgorithm.py` contains a simple text-area demo.
- `src/` - Source code for implementations, helpers, and modules used by the pages. Each homework or exercise can get its own subfolder. Example: `src/1_HillClimbingAlgorithm/main.py` contains a simple module for the hill climbing exercise.
- `README.md` - This file.
- `pyproject.toml`, `uv.lock` - project metadata / locks.

Homework/schema guideline

When you add a new homework/exercise, follow this schema so the repository stays consistent and easy to navigate:

- Add one Streamlit page in `pages/` named with a leading numeric index and a short descriptive name, e.g. `1_HillClimbingAlgorithm.py`, `2_ParticleSwarm.py`.
- Add a corresponding folder under `src/` with the same numeric prefix and descriptive name, e.g. `src/1_HillClimbingAlgorithm/`.
- Place implementation modules (library code, algorithms, helpers) inside that `src/<exercise>/` folder. Provide a small `main.py` or other modules that the Streamlit page can import.
- Place data files (if any) inside a `data/` subfolder under the exercise folder, e.g. `src/2_ParticleSwarm/data/`.
- Keep the Streamlit page file minimal: UI elements, parameter inputs, and imports from the matching `src/` package. Avoid putting heavy algorithm code directly in `pages/` files.

This pattern keeps UI and logic separated and makes it easy to add, test, and reuse implementations.

## Naming conventions

To keep files discoverable and ordered, use the following naming conventions for Streamlit pages and source folders/files:

1. Numeric prefix: Start exercise folders and page filenames with a numeric prefix followed by an underscore, e.g. `1_`, `2_`, `10_`. This ensures pages are ordered in the Streamlit sidebar and folders are easy to read.
2. Descriptive PascalCase for names: After the numeric prefix, use a concise PascalCase descriptive name without spaces, e.g. `HillClimbingAlgorithm`, `ParticleSwarm`. Combine with the numeric prefix like `3_ParticleSwarm.py` and `src/3_ParticleSwarm/`.
3. Module names: Keep module filenames lowercase with underscores if they implement smaller components (e.g. `utils.py`, `evaluation.py`). Public entry points can be `main.py` or a package `__init__.py` when needed.
4. Keep UI and logic separated: Pages in `pages/` should not contain large algorithm implementations. Instead they should import from `src/` so tests and automation can import algorithm code without starting Streamlit.

Example mapping from this repository

- `pages/1_HillClimbingAlgorithm.py` → `src/1_HillClimbingAlgorithm/main.py`

## Running the app

To start the Streamlit app from the repository root (folder named `SwarmIntelligenceAndEvolutionaryAlgorithms`), run the following command:

```bash
uv run streamlit run streamlit_app.py
```

This will launch Streamlit and open the multi-page app defined by the files in `pages/` and the `streamlit_app.py` entry point.

