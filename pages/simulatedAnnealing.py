import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from src.SimulatedAnnealing.tsp_algorithm import tsp, get_sa_route_coords, total_distance, simulated_annealing
import matplotlib.pyplot as plt


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
        "results": "Stadt wählen und die beste Route mit dem TSP-Algorithmus berechnen.",
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
        "results": "Select city and compute the best route using the TSP algorithm.",
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
    # Lade Distanzmatrix
    dist_matrix = tsp.distance
    city_names = tsp.get_all_names()
    n = len(city_names)

    st.title("Distanzmatrix der Städte")

    # Heatmap plotten
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(dist_matrix, cmap='viridis')

    # Achsenbeschriftung
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(city_names, rotation=90, fontsize=8)
    ax.set_yticklabels(city_names, fontsize=8)

    # Farbskala
    fig.colorbar(cax)

    st.pyplot(fig)
# =====================================================
# TAB 2: RESULTS / ERGEBNISSE
# =====================================================
with tabs[2]:
    st.info(T[lang]["results"])

    # --- Städte laden ---
    city_names = tsp.get_all_names()
    start_city = st.selectbox("Startstadt", city_names)

    # --- Sidebar-Hyperparameter ---
    st.sidebar.header("SA-Hyperparameter")
    T_start = st.sidebar.slider("Starttemperatur (T_start)", 100, 10000, 2000, step=100)
    T_end = st.sidebar.slider("Endtemperatur (T_end)", 0.1, 50.0, 1.0)
    alpha = st.sidebar.slider("Abkühlrate (alpha)", 0.95, 0.999, 0.995, step=0.001)
    max_iter = st.sidebar.slider("Max. Iterationen", 1000, 100000, 20000, step=1000)
    reheating_factor = st.sidebar.slider("Reheating-Faktor", 1.0, 2.0, 1.1, step=0.1)

    if st.button("Run"):
        start_index = tsp.get_city_index(start_city)

        # --- TSP berechnen ---
        best_route, best_distance, history = simulated_annealing(
            tsp.distance,
            start_city_index=start_index,
            T_start=T_start,
            T_end=T_end,
            alpha=alpha,
            max_iter=max_iter,
            reheating_factor=reheating_factor,
            stagnation_limit=2500,
            return_history=True
        )

        # Gesamtdistanz
        from src.SimulatedAnnealing.tsp_algorithm import total_distance
        total_dist = total_distance(best_route, tsp.distance)

        # Gefundene Route anzeigen
        st.subheader("Gefundene Route:")
        for i, idx in enumerate(best_route):
            st.text(f"{i+1}: {tsp.city_names[idx]}")
        st.text(f"{len(best_route)+1}: {tsp.city_names[best_route[0]]} (Rückkehr)")
        st.success(f"Gesamtdistanz: {total_dist/1000:.2f} km")

        st.subheader("Verbesserungsverlauf")
        # History in km
        history_km = [d / 1000 for d in history]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=history_km,
            mode='lines+markers',
            name='Beste Distanz',
            line=dict(color='red'),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title="Verbesserungsverlauf der SA",
            xaxis_title="Schritte",
            yaxis_title="Distanz [km]",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Koordinaten für finale Route (inkl. Rückkehr zur Startstadt) ---
        full_route = best_route + [best_route[0]]
        sa_coords = get_sa_route_coords(full_route, tsp)

        # --- Karte laden ---
        map_img = Image.open("src/SimulatedAnnealing/landscape/austria_map.png").convert("RGBA")
        draw = ImageDraw.Draw(map_img)
        width, height = map_img.size

        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = None

        # --- Projektionsfunktion ---
        def project(coord):
            lon, lat = coord
            left, right = 9.25, 17.25
            top, bottom = 49.25, 46.25
            x = int((lon - left) / (right - left) * width)
            y = int((top - lat) / (top - bottom) * height)
            return x, y

        # --- Linien der Route zeichnen ---
        for i in range(len(sa_coords)-1):
            draw.line([project(sa_coords[i]), project(sa_coords[i+1])], fill=(255,0,0,255), width=3)

        # --- Städtepunkte und Namen ---
        for i, idx in enumerate(best_route):
            city = tsp.city_names[idx]
            x, y = project(tsp.get_city_coord(city))
            color = (0,255,0,255) if i == 0 else (0,0,255,255)  # Startstadt grün
            draw.ellipse([x-5, y-5, x+5, y+5], fill=color)
            if font:
                draw.text((x+6, y-6), city, fill=(0,0,0,255), font=font)
            else:
                draw.text((x+6, y-6), city, fill=(0,0,0,255))

        st.subheader(f"Finale Route ab {start_city}")
        st.image(map_img)



# =====================================================
# TAB 3: DISCUSSION / DISKUSSION
# =====================================================
with tabs[3]:
    st.markdown(T[lang]["discussion"])
