import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

from src.SimulatedAnnealing.tsp_algorithm import tsp, get_route_coords


# Streamlit config
st.set_page_config(page_title="Simulated Annealing - TSP", layout="wide")

# ---------------------------
# Sprachumschaltung (EN / DE)
# ---------------------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "DE"

# Sprachoptionen mit Flaggen
lang_options = {
    "🇩🇪 Deutsch": "DE",
    "🇬🇧 English": "EN"
}

col_lang1, col_lang2 = st.columns([1, 6])
with col_lang1:
    # Setze aktuellen Key als default_value, damit es stateful ist
    selected_flag = st.selectbox(
        "Sprache / Language",
        options=list(lang_options.keys()),
        key="lang_selectbox"
    )
    # State sofort setzen
    st.session_state["lang"] = lang_options[selected_flag]

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
        "results": "2 Städte wählen und die beste Route mit dem TSP-Algorithmus berechnen.",
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
        "results": "Select 2 cities and compute the best route using the TSP algorithm.",
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

    # Auswahlfelder für Start und Ziel
    city_names = tsp.get_all_names()
    start_city = st.selectbox("Startstadt", city_names)
    end_city = st.selectbox("Zielstadt", city_names, index=1)

    if st.button("Route berechnen"):
        if start_city == end_city:
            st.warning("Start- und Zielstadt dürfen nicht gleich sein!")
        else:
            # Geo-Koordinaten der Route aus Algorithmus holen
            coords = get_route_coords(start_city, end_city)
            if not coords:
                st.error("Keine Route gefunden!")
            else:
                # Karte laden
                map_img = Image.open("src/SimulatedAnnealing/landscape/austria_map.png").convert("RGBA")
                draw = ImageDraw.Draw(map_img)
                width, height = map_img.size

                # Geo → Pixel Mapping
                def project(coord):
                    lon, lat = coord
                    # Österreich: west=9.25, east=17.25, north=49, south=46.2
                    left, right = 9.25, 17.25
                    top, bottom = 49.25, 46.25
                    x = int((lon - left) / (right - left) * width)
                    y = int((top - lat) / (top - bottom) * height)
                    return x, y

                # Route zeichnen
                for i in range(len(coords) - 1):
                    draw.line([project(coords[i]), project(coords[i + 1])], fill=(255, 0, 0, 255), width=3)

                # Start & Ziel markieren
                start_px = project(coords[0])
                end_px = project(coords[-1])
                r = 6  # Radius
                draw.ellipse([start_px[0]-r, start_px[1]-r, start_px[0]+r, start_px[1]+r], fill="green")
                draw.ellipse([end_px[0]-r, end_px[1]-r, end_px[0]+r, end_px[1]+r], fill="blue")

                # Karte anzeigen
                st.image(map_img, caption=f"Route von {start_city} nach {end_city}")

# =====================================================
# TAB 3: DISCUSSION / DISKUSSION
# =====================================================
with tabs[3]:
    st.markdown(T[lang]["discussion"])
