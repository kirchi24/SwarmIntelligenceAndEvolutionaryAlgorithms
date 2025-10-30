import streamlit as st
import numpy as np

# Streamlit config
st.set_page_config(page_title="Simulated Annealing - TSP", layout="wide")

# ---------------------------
# Sprachumschaltung (EN / DE)
# ---------------------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "DE"

def toggle_lang():
    st.session_state["lang"] = "EN" if st.session_state["lang"] == "DE" else "DE"

col_lang1, col_lang2 = st.columns([1, 6])
with col_lang1:
    st.button(
        "EN English" if st.session_state["lang"] == "DE" else "DE Deutsch",
        on_click=toggle_lang
    )

lang = st.session_state["lang"]
# Textdictionary
T = {
    "DE": {
        "title": "Simulated Annealing - Travelling Salesman Problem",
        "tabs": ["Einführung", "Methoden", "Ergebnisse", "Diskussion"],
        "intro": """
        Diese App dokumentiert den **Simulated Annealing Algorithmus** für das Travelling-Salesman-Problem (TSP).
        Ziel ist es, alle Städte **einmal zu besuchen** und die Gesamtdistanz zu minimieren.
        """,
        "methods": """
        **Lösungsdarstellung:** Permutation der Städte.  
        **Energiefunktion:** Gesamtdistanz der Route.  
        **Nachbarschaftserzeugung:** Inversion oder Swap zweier Städte.  
        **Kühlstrategie:** Linear, exponentiell oder adaptiv.  
        **Parameter:** Starttemperatur, Abkühlrate, Iterationen.
        """,
        "results": "Die Ergebnisse werden später angezeigt, sobald der Algorithmus implementiert ist.",
        "discussion": """
        **Diskussion**  
        - SA kann lokale Minima besser verlassen als gierige Algorithmen.  
        - Qualität hängt von Kühlstrategie, Nachbarschaftserzeugung und Starttemperatur ab.  
        - Visualisierung der Route zeigt Konvergenzverhalten.  
        - Einschränkungen: große Stadtmengen, Parametertuning entscheidend.
        """
    },
    "EN": {
        "title": "Simulated Annealing - Travelling Salesman Problem",
        "tabs": ["Introduction", "Methods", "Results", "Discussion"],
        "intro": """
        This page documents the **Simulated Annealing algorithm** for the Travelling Salesman Problem (TSP). 
        The goal is to visit all cities **exactly once** and minimize total travel distance.
        """,
        "methods": """
        **Solution representation:** permutation of cities.  
        **Energy function:** total route distance.  
        **Neighbor generation:** inversion or swap of two cities.  
        **Cooling schedule:** linear, exponential, or adaptive.  
        **Parameters:** initial temperature, cooling rate, iterations.
        """,
        "results": "Results will be displayed here once the algorithm is implemented.",
        "discussion": """
        **Discussion**  
        - SA can escape local minima better than greedy algorithms.  
        - Solution quality depends on cooling schedule, neighbor generation, and initial temperature.  
        - Route visualization shows convergence behavior.  
        - Limitations: large city sets, parameter tuning is crucial.
        """
    }
}

st.title(T[lang]["title"])

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(T[lang]["tabs"])

# =====================================================
# TAB 0: INTRODUCTION / EINLEITUNG
# =====================================================
with tabs[0]:
    st.markdown(T[lang]["intro"])

# =====================================================
# TAB 1: METHODS / METHODEN
# =====================================================
with tabs[1]:
    st.markdown(T[lang]["methods"])

# =====================================================
# TAB 2: RESULTS / ERGEBNISSE
# =====================================================
with tabs[2]:
    st.info(T[lang]["results"])

# =====================================================
# TAB 3: DISCUSSION / DISKUSSION
# =====================================================
with tabs[3]:
    st.markdown(T[lang]["discussion"])
