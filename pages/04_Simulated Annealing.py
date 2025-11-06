import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.SimulatedAnnealing.tsp_algorithm import tsp, get_sa_route_coords, total_distance, simulated_annealing
from src.SimulatedAnnealing.plot_route import plot_route
import matplotlib.pyplot as plt
from io import BytesIO

# Streamlit config
st.set_page_config(page_title="Simulated Annealing - TSP", layout="wide")

# Sprachumschaltung (EN / DE)

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
        "tabs": ["Einführung", "Methoden", "TSPlight", "TSPfull", "Diskussion"],
        "intro": """
        Diese App dokumentiert den **Simulated Annealing Algorithmus** für das Travelling-Salesman-Problem (TSP).
        Ziel ist es, alle Städte **einmal zu besuchen** und die Gesamtdistanz zu minimieren.
        """,
        "methods": """
        """,
        "results": "Stadt wählen und die beste Route mit dem TSP-Algorithmus berechnen.",
        "TSPlight": "Leichte Version des TSP zur schnellen Visualisierung der Route zwischen ausgewählten Städten.",
        "TSPfull": "Vollversion des TSP mit allen Städten und vollständiger Optimierung.",
        "discussion": """
        **Diskussion**  
        - SA kann lokale Minima besser verlassen als gierige Algorithmen.  
        - Qualität hängt von Kühlstrategie, Nachbarschaftserzeugung und Starttemperatur ab.   
        - Die **Kühlstrategie** hat einen entscheidenden Einfluss auf die Qualität der Lösung und die Konvergenzgeschwindigkeit.  
        - Eine **langsame Abkühlung** (z. B. geringe Reduktion der Temperatur pro Iteration) ermöglicht es, länger nach besseren Lösungen zu suchen und lokale Minima zu vermeiden – die Laufzeit steigt jedoch deutlich.  
        - Eine **schnelle Abkühlung** führt zu kürzeren Berechnungszeiten, kann aber dazu führen, dass das Verfahren zu früh „einfriert“ und in suboptimalen Lösungen steckenbleibt.  
        - Ein gutes Gleichgewicht zwischen **Exploration** (Suche in neuen Gebieten der Lösungslandschaft) und **Exploitation** (Verfeinerung guter Lösungen) ist entscheidend.  
        - Die Kühlstrategie beeinflusst somit direkt, wie effektiv der Algorithmus zwischen diesen Phasen wechselt.  
        - Für große Städtemengen ist eine adaptive oder problemabhängige Kühlrate oft vorteilhaft.  
        - Insgesamt bestimmt die Wahl der Kühlstrategie maßgeblich, ob der Algorithmus eine nahezu optimale Route findet oder nur eine akzeptable Näherung erreicht.


        """
    },
    "EN": {
        "title": "Simulated Annealing - Travelling Salesman Problem",
        "tabs": ["Introduction", "Methods", "TSPlight", "TSPfull",  "Discussion"],
        "intro": """
        This page documents the **Simulated Annealing algorithm** for the Travelling Salesman Problem (TSP). 
        The goal is to visit all cities **exactly once** and minimize total travel distance.
        """,
        "methods": """
        """,
        "results": "Select city and compute the best route using the TSP algorithm.",
        "TSPlight": "Light version of the TSP for quick visualization of the route between selected cities.",
        "TSPfull": "Full TSP version with all cities and complete optimization.",
        "discussion": """
        **Discussion**  
        - Simulated Annealing (SA) can escape local minima more effectively than greedy algorithms.  
        - Solution quality depends on the cooling schedule, neighborhood generation, and initial temperature.  
        - The **cooling strategy** has a decisive impact on both the solution quality and the convergence speed.  
        - A **slow cooling** schedule (e.g., small temperature reduction per iteration) allows the algorithm to explore longer and avoid local minima, but it significantly increases runtime.  
        - A **fast cooling** schedule reduces computation time but may cause the algorithm to "freeze" too early, resulting in suboptimal solutions.  
        - Finding a good balance between **exploration** (searching new regions of the solution space) and **exploitation** (refining good solutions) is essential.  
        - The cooling strategy directly determines how effectively the algorithm transitions between these two phases.  
        - For large city sets, adaptive or problem-specific cooling rates are often beneficial.  
        - Ultimately, the choice of cooling strategy largely determines whether the algorithm finds a near-optimal route or merely an acceptable approximation.
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
    if lang == "DE":
        st.markdown(
            """
            - Simulated Annealing (SA) ist ein **stochastischer Optimierungsalgorithmus**, 
            der von der Abkühlung fester Stoffe inspiriert ist.  
            - Zu Beginn ist die "Temperatur" hoch - das System akzeptiert auch schlechtere Lösungen, 
            um **lokale Minima zu vermeiden**.  
            - Mit sinkender Temperatur werden nur noch **bessere oder leicht schlechtere** Lösungen akzeptiert, 
            bis das System stabil wird.  
            - So lässt sich eine **annähernd optimale Lösung** für kombinatorische Probleme wie das 
            **Travelling Salesman Problem (TSP)** finden.
            """
        )
    else:
        st.markdown(
            """
            - Simulated Annealing (SA) is a **stochastic optimization algorithm** inspired by the cooling process of metals.  
            - At high "temperature", the algorithm accepts even worse solutions to **escape local minima**.  
            - As the temperature decreases, it becomes more selective, focusing on improving or slightly worse moves.  
            - This gradual cooling helps find a **near-optimal solution** for combinatorial problems such as the 
            **Travelling Salesman Problem (TSP)**.
           """
        )    
# =====================================================
# TAB 1: METHODS / METHODEN
# =====================================================
with tabs[1]:
    if lang == "DE":
        st.markdown("""
        ## Simulated Annealing Algorithmus für das Travelling Salesman Problem (TSP)

        ### 1. Startlösung
        - Der Algorithmus startet mit einer **Nearest-Neighbor-Heuristik** (`nearest_neighbor_solution()`), beginnend bei einer fixierten Startstadt (`start_city_index`).
        - Diese Heuristik baut eine erste Rundreise auf, indem sie stets die **nächste unbesuchte Stadt** wählt.  
        - Die Gesamtdistanz dieser Startlösung wird als **aktuelle Lösung** und zugleich als **beste bisher gefundene Lösung** gesetzt.


        ### 2. Nachbarschaftserzeugung
        Drei verschiedene Operationen erzeugen alternative Routen:  
        - In jeder Iteration wird eine **neue Nachbarroute** aus der aktuellen Route erzeugt:
        - Standardmäßig durch `get_neighbor(current_route)`, das zufällig eine der lokalen Änderungen auswählt (z. B. Swap, Reinsertion, 2-Opt).
        - Falls sich die Lösung **über längere Zeit nicht verbessert** (mehr als `0.75 * stagnation_limit` Iterationen ohne Verbesserung), wird gezielt der **2-Opt-Operator** eingesetzt (`neighborhood_boost=True`), um die Suche lokal zu intensivieren.

        Eine dieser Operationen wird zufällig entsprechend vordefinierter Wahrscheinlichkeiten ausgewählt.  

        ### 2.2. Verhalten und Lokalität von 2-Opt
        2-Opt invertiert ein Segment der Route, wodurch einige Verbindungen länger oder kürzer werden können. Jede Operation verändert **nur zwei Kanten**, behält die Gesamtstruktur der Route bei und hat dennoch starke Wirkung - ideal für die Intensivierung der lokalen Suche.

        ### 3. Energiefunktion
        - Bewertet die Qualität einer Route anhand der **Gesamtdistanz**.
        - Ungültige oder unendliche Werte werden als „unendlich“ behandelt und verworfen.

        ### 4. Akzeptanzkriterium
        - **Bessere Lösungen** werden immer akzeptiert.
        - **Schlechtere Lösungen** werden mit Wahrscheinlichkeit $P = \\exp(-\\Delta / T)$ akzeptiert (Metropolis-Kriterium).
        - So kann der Algorithmus lokale Minima verlassen.

        ### 5. Kühlstrategie
        - **Exponentielle Abkühlung:** $T_{neu} = \\alpha \\cdot T_{alt}$
        - **Dynamisches Reheating:** Falls über viele Iterationen keine Verbesserung erfolgt, wird die Temperatur kurzfristig erhöht, um neue Bereiche zu durchsuchen.
                    
        ### 6. Stoppkriterien
        Der Algorithmus endet, wenn eine der folgenden Bedingungen erfüllt ist:
        1. **Temperatur zu niedrig:** $ T < T_{\\text{end}} $
        2. **Maximale Iterationen:** $ \\text{iteration\_count} > \\text{max\_iter} $
        
        Das jeweilige **Abbruchkriterium** wird im Code als `stopping_reason` gespeichert.


        ### 7. Rückgabe
        - **Beste gefundene Route** (`best_route`)
        - **Minimale Gesamtdistanz** (`best_distance`)
        - Optional: **Historie der besten Distanzen pro Iteration** (`return_history=True`)

        ---  

        ### Implementierungsdetails
        - `total_distance(route, dist_matrix)` → Berechnet Gesamtdistanz einer Route.
        - `nearest_neighbor_solution(dist_matrix, start)` → Erzeugt Startlösung.
        - `two_opt()`, `reinsertion()`, `swap()` → Nachbarschaftsoperatoren.
        - `simulated_annealing()` → Führt den gesamten Optimierungsprozess aus.
        """)
    else:
        st.markdown("""
        ## Simulated Annealing Algorithm for the Travelling Salesman Problem (TSP)

        ### 1. Initial Solution
        - A greedy **Nearest-Neighbor** solution is generated, starting from a chosen city.
        - The route visits all cities once and returns to the starting point.

        ### 2. Neighborhood Generation
        Three different operations create alternative routes:
        - **2-Opt:** Selects two cut points and reverses the segment between them. This changes only two route connections, keeping the rest of the path intact.
        - **Reinsertion:** Removes one city and reinserts it at another position in the route. Produces medium-sized local changes that help escape shallow local minima.
        - **Swap:** Exchanges the positions of two cities. Generates very small, quick variations and increases randomness early in the search.
        One of these operations is chosen randomly uniformly.
                    
        ### 2.2. 2-Opt Behavior and Locality
        2-Opt reverses a route segment, which can make some connections longer or shorter. Each move changes **only two edges**, keeping the overall tour structure intact while still having a strong impact, making it ideal for local search intensification.

        ### 3. Energy Function
        - Evaluates the total **distance** of the route.
        - Invalid or infinite values are treated as infinite (rejected).

        ### 4. Acceptance Criterion
        - **Better solutions** are always accepted.
        - **Worse solutions** are accepted with probability $P = \\exp(-\\Delta / T) $ (Metropolis criterion).
        - This allows the algorithm to escape local minima.

        ### 5. Cooling Schedule
        - **Exponential cooling:**  ($T_{neu} = \\alpha \\cdot T_{alt}$)
        - **Dynamic reheating:** If stagnation occurs for many iterations,
          temperature is temporarily increased to escape local minima.

        ### 6. Stopping Criteria
        - Final temperature $ T_{end} $ reached  
        - Maximum number of iterations exceeded

        ### 7. Output
        - Best route found  
        - Total distance  
        ---  

        ### Implementation Details 
        - `total_distance(route, dist_matrix)` → Computes total route distance.
        - `nearest_neighbor_solution(dist_matrix, start)` → Builds initial greedy route.
        - `two_opt()`, `reinsertion()`, `swap()` → Neighborhood operators.
        - `simulated_annealing()` → Executes the full optimization process.
        """)


        # -------------------
    # DISTANZMATRIX ANZEIGEN
    # -------------------
    st.markdown(T[lang]["methods"])

    dist_matrix = tsp.distance
    city_names = tsp.get_all_names()
    n = len(city_names)

    st.subheader("Distanzmatrix der Städte" if lang == "DE" else "Distance Matrix of Cities")

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(dist_matrix, cmap='viridis')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(city_names, rotation=90, fontsize=8)
    ax.set_yticklabels(city_names, fontsize=8)

    fig.colorbar(cax)
    st.pyplot(fig)

    st.markdown("---")

    if lang == "DE":
        st.markdown("""
        - Mit der obigen Distanzmatrix können wir die Entfernungen zwischen den Städten ablesen.
        - Der Simulated Annealing Algorithmus nutzt diese Matrix, um die Gesamtdistanz einer Route zu berechnen.
        - Hiermit wurde überprüft, ob zwischen allen Städten eine Verbindung besteht und die Distanzen plausibel sind.
        """)
    else:
        st.markdown("""
        - The above distance matrix shows the distances between cities.
        - The Simulated Annealing algorithm uses this matrix to compute the total distance of a route.
        - This matrix was verified to ensure all cities are connected and distances are plausible.
        """)


# =====================================================
# TAB 2: TSP light
# =====================================================
with tabs[2]:
    st.markdown(T[lang]["TSPlight"])
    
    # --- Städte laden ---
    city_names = tsp.get_all_names()

    # --- Benutzer wählt beliebig viele Städte ---
    selected_cities = st.multiselect(
        "Wähle die Route (beliebig viele Städte)", 
        options=city_names,
        default=city_names[0:3],
        key="selected_cities_light"
    )

    if len(selected_cities) < 3:
        st.warning(
            "Bitte wähle mindestens 3 Städte für eine Route aus." 
            if lang=="DE" else "Please select at least 3 cities for a route."
        )
    else:
        start_city = selected_cities[0]

        # ---- Hyperparameter ---
        st.header("SA-Hyperparameter (Light)")
        T_start = st.slider("Starttemperatur (T_start)", 100, 10000, 2000, step=100, key="T_start_light")
        T_end = st.slider("Endtemperatur (T_end)", 0.1, 50.0, 1.0, key="T_end_light")
        alpha = st.slider("Abkühlrate (alpha)", 0.95, 1.0, 0.995, step=0.001, key="alpha_light")
        max_iter = st.slider("Max. Iterationen", 1000, 25000, 10000, step=1000, key="max_iter_light")
        reheating_factor = st.slider("Reheating-Faktor", 1.0, 2.0, 1.1, step=0.1, key="reheat_light")
        stagnation_limit = st.slider("Stagnationslimit", 500, 10000, 2500, step=250, key="stagnation_light")
        neighborhood_boost = st.checkbox("Nachbarschaftsboost aktivieren", value=True, key="neighborhood_boost_light")

        distance_objective = st.selectbox(
            "Wähle dein Ziel:",
            ["Distanz", "Dauer", "Schritte"],
            index=0,
            key="distance_objective_light"
        )

        if st.button("Run TSP Light", key="run_light"):
            # --- Nur die gewählten Städte berücksichtigen ---
            selected_indices = [tsp.get_city_index(city) for city in selected_cities]

            # --- Zielmetrik bestimmen ---
            selected_metric = (
                tsp.distance if distance_objective == "Distanz" else
                tsp.duration if distance_objective == "Dauer" else
                tsp.steps_count
            )

            # --- Reduzierte Distanzmatrix erstellen ---
            light_distance_matrix = np.array([
                [selected_metric[i, j] for j in selected_indices]
                for i in selected_indices
            ])

            # --- Startindex in der REDUZIERTEN Matrix bestimmen ---
            # (immer die erste ausgewählte Stadt)
            start_city_index = 0

            # --- Simulated Annealing mit NUR ausgewählten Städten ---
            best_route, best_distance, stopping_reason, history, iterations = simulated_annealing(
                dist_matrix=light_distance_matrix,
                start_city_index=start_city_index,
                T_start=T_start,
                T_end=T_end,
                alpha=alpha,
                max_iter=max_iter,
                reheating_factor=reheating_factor,
                stagnation_limit=stagnation_limit,
                return_history=True,
                neighborhood_boost=neighborhood_boost
            )   

            # select found route and normalize names
            route_cities = [selected_cities[idx].replace(" ", "_") for idx in best_route]

            st.subheader("Gefundene Route:")
            for i, city in enumerate(route_cities):
                st.text(f"{i+1}: {city}")
            st.text(f"{len(route_cities)+1}: {route_cities[0]} (Rückkehr)")
            st.success(
                f"Gesamtdistanz: {best_distance:.1f} - "
                f"Stoppgrund: {stopping_reason} - "
                f"Iterationen: {iterations}"
            )
            # --- Verbesserungsverlauf plotten ---
            if history:
                import plotly.graph_objects as go
                history_km = [d for d in history]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(history_km))),
                    y=history_km,
                    mode='lines+markers',
                    name=f'Beste{distance_objective}',
                    line=dict(color='red'),
                    marker=dict(size=2)
                ))
                fig.update_layout(
                    title="Verbesserungsverlauf der SA",
                    xaxis=dict(title="Schritte", range=[0, len(history_km)]),
                    yaxis=dict(title=distance_objective, range=[0, max(history_km)]),
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- Finale Route direkt mit plot_route anzeigen ---
            st.subheader(f"Finale Route ab {start_city}")
            import matplotlib.pyplot as plt
            from io import BytesIO
            from src.SimulatedAnnealing.plot_route import plot_route

            # plot_route erzeugt die Matplotlib-Figur
            plot_route(route_cities)  

            # In BytesIO speichern und in Streamlit anzeigen
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.image(buf)
            plt.close()
# =====================================================
# TAB 3: TSP full
# =====================================================
with tabs[3]:
    st.markdown(T[lang]['TSPfull'])

    # --- Städte laden ---
    city_names = tsp.get_all_names()
    start_city = st.selectbox("Startstadt", city_names)

    # --- Sidebar-Hyperparameter ---
    st.header("SA-Hyperparameter")
    T_start = st.slider("Starttemperatur (T_start)", 100, 10000, 2000, step=100)
    T_end = st.slider("Endtemperatur (T_end)", 0.1, 50.0, 1.0)
    alpha = st.slider("Abkühlrate (alpha)", 0.95, 1.0, 0.995, step=0.001)
    max_iter = st.slider("Max. Iterationen", 1000, 25000, 20000, step=1000)
    reheating_factor = st.slider("Reheating-Faktor", 1.0, 2.0, 1.1, step=0.1)
    stagnation_limit = st.slider("Stagnationslimit", 500, 10000, 2500, step=250)
    neighborhood_boost = st.checkbox("Nachbarschaftsboost aktivieren", value=True, key="neighborhood_boost_full")
    distance_objective = st.selectbox(
        "Wähle dein Ziel:",
        ["Distanz", "Dauer", "Schritte"],
        index=0,
        key="distance_objective_full"
    )

    if st.button("Run"):
        start_index = tsp.get_city_index(start_city)

        selected_metric = tsp.distance if distance_objective == "Distanz" else (
            tsp.duration if distance_objective == "Dauer" else tsp.steps_count
        )

        # --- TSP berechnen ---
        best_route, best_distance, stopping_reason, history, iterations = simulated_annealing(
            selected_metric,
            start_city_index=start_index,
            T_start=T_start,
            T_end=T_end,
            alpha=alpha,
            max_iter=max_iter,
            reheating_factor=reheating_factor,
            stagnation_limit=stagnation_limit,
            return_history=True,
            neighborhood_boost=neighborhood_boost
        )

        # Gesamtdistanz
        from src.SimulatedAnnealing.tsp_algorithm import total_distance
        total_dist = total_distance(best_route, tsp.distance)

        # Gefundene Route anzeigen
        st.subheader("Gefundene Route:")
        for i, idx in enumerate(best_route):
            st.text(f"{i+1}: {tsp.city_names[idx]}")
        st.text(f"{len(best_route)+1}: {tsp.city_names[best_route[0]]} (Rückkehr)")
        st.success(
            f"Gesamtdistanz: {best_distance:.1f} - "
            f"Stoppgrund: {stopping_reason} - "
            f"Iterationen: {iterations}"
        )

        st.subheader("Verbesserungsverlauf")
        # History in km
        history_km = [d for d in history]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(history_km))),
            y=history_km,
            mode='lines+markers',
            name=f'Beste{distance_objective}',
            line=dict(color='red'),
            marker=dict(size=2)
        ))

        fig.update_layout(
            title="Verbesserungsverlauf der SA",
            xaxis=dict(title="Schritte", range=[0, len(history_km)]),
            yaxis=dict(title=distance_objective, range=[0, max(history_km)]),
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
with tabs[4]:
    st.markdown(T[lang]["discussion"])

