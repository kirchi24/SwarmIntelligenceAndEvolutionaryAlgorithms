# test_plot.py im Projekt-Root
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.SimulatedAnnealing.plot_route import plot_route

example_route = ["Amstetten", "Dornbirn", "Graz", "Amstetten"]
plot_route(example_route)