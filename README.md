# Swarm Intelligence and Evolutionary Algorithms

This repository is developed as part of the Swarm Intelligence and Evolutionary Algorithms course in the Data Science and Artificial Intelligence program at FH Joanneum Graz.

It contains various tasks and experiments implementing swarm intelligence and evolutionary algorithms, which are documented and interactively demonstrated using Streamlit. The goal is to make the functionality and behavior of these algorithms understandable and visually accessible.

## Project structure

The repository follows a simple layout to separate Streamlit pages (user-facing exercises/demos) from reusable source code and other project files. Current top-level files and folders:

- `home.py` - Streamlit app entry point and app configuration.
- `pages/` - Streamlit multi-page files. Each page corresponds to an exercise or demo and is loaded by Streamlit when the app runs. Example: `pages/hillClimbingAlgorithm.py` contains the hill climbing demo.
- `src/` - Source code for implementations, helpers, and modules used by the pages. Each homework or exercise can get its own subfolder. Example: `src/HillClimbingAlgorithm/algorithm.py` contains the hill climbing implementation.
- `README.md` - This file.
- `pyproject.toml`, `uv.lock` - project metadata / locks.

Homework/schema guideline

When you add a new homework/exercise, follow this schema so the repository stays consistent and easy to navigate:

- Add one Streamlit page in `pages/` with a clear, descriptive name, e.g. `hillClimbingAlgorithm.py`, `ParticleSwarm.py`.
- Add a corresponding folder under `src/` with a Python-friendly package name in snakeCase, e.g. `src/HillClimbingAlgorithm/`.
- Place implementation modules (library code, algorithms, helpers) inside that `src/<exercise>/` folder. Prefer `algorithm.py` or another clear module name the Streamlit page can import.
- Keep the Streamlit page file minimal: UI elements, parameter inputs, and imports from the matching `src/` package. Avoid putting heavy algorithm code directly in `pages/` files.
- Place data files (if any) inside a `data/` subfolder under the exercise folder, e.g. `src/particle_swarm/data/`.
- Place the description of the exercise (the `.pdf` file) inside that `src/<exercise>/` folder as well and name it `exercise_description.pdf`.

This pattern keeps UI and logic separated and makes it easy to add, test, and reuse implementations.

## Naming conventions

To keep files discoverable and Python-friendly, use the following naming conventions for Streamlit pages and source folders/files:

1. Descriptive names: Use concise, descriptive names for pages, e.g. `hillClimbingAlgorithm.py`, `ParticleSwarm.py` (PascalCase for page files works well in the Streamlit UI).
2. Python packages: Use snakeCase for module filenames in `src/`, e.g. `src/HillClimbingAlgorithm/algorithm.py`.
3. Module names: Keep module filenames lowercase with underscores (e.g. `utils.py`, `evaluation.py`). Public entry points can be `algorithm.py`, `main.py`, or a package `__init__.py` when needed.
4. Keep UI and logic separated: Pages in `pages/` should not contain large algorithm implementations. Instead they should import from `src/` so tests and automation can import algorithm code without starting Streamlit.

Example mapping from this repository

- `pages/hillClimbingAlgorithm.py` → `src/HillClimbingAlgorithm/algorithm.py`

## Running the app

To start the Streamlit app from the repository root (folder named `SwarmIntelligenceAndEvolutionaryAlgorithms`), run the following command:

```bash
uv run streamlit run home.py
```

This will launch Streamlit and open the multi-page app defined by the files in `pages/` and the `home.py` entry point.

