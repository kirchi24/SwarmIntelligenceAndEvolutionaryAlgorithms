import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tempfile
import time
from io import BytesIO

from src.Differential_Evolution.differential_evolution import (
    differential_evolution,
)
from src.Differential_Evolution.radial_encoding import (
    radii_to_polygon,
    place_polygon_against_corridor,
)
from src.Differential_Evolution.utils import (
    construct_corridor,
    move_and_rotate_smooth,
    animate_shape,
)


# ========================================================
# Streamlit Page Setup
# ========================================================

st.set_page_config(
    page_title="Differential Evolution - Moving Sofa Problem",
    layout="wide"
)

st.title("Moving Sofa Problem - Differential Evolution Optimizer")


# ========================================================
# Tabs
# ========================================================

tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])


# ========================================================
# TAB 1 - INTRODUCTION
# ========================================================
with tabs[0]:
    st.markdown("## Einführung")

    st.markdown(
        r"""
Das **Moving-Sofa-Problem** ist ein klassisches geometrisches Optimierungsproblem:

Finde die Form maximaler Fläche, die um eine 90 Grad - Kante eines L-Korridors bewegt werden kann.


Dieses Projekt verwendet **Differential Evolution (DE)** zur Optimierung der Form, kodiert über eine radiale Repräsentation $r (\theta)$.

---

### Bekannte Lösung (Hammersley, 1968)

Die aktuell beste bekannte Lösung hat eine Fläche:

$$
A \approx 2.2074 \, \text{Einheiten}^2
$$

Diese Lösung ist stark konkav und analytisch schwierig herzuleiten.

---

### Ziel dieser Anwendung
- Parameter der Differential Evolution untersuchen  
- L-Korridormaße konfigurieren  
- Bewegung des Sofas animieren  
- Einfluss der Parameter verstehen  
"""
    )


# ========================================================
# TAB 2 - METHODS
# ========================================================
with tabs[1]:
    st.markdown("## Methoden")

    st.markdown(
        r"""
### 1. Radiale Repräsentation

Eine Form wird durch $(K)$ Radien beschrieben:

$$
r = [r_1, r_2, \ldots, r_K]
$$

Die zugehörigen Winkel:

$$
\theta_k = \frac{2\pi k}{K}
$$

Die Punkte im 2D-Raum:

$$
P_k = (r_k \cos \theta_k,\; r_k \sin \theta_k)
$$

---

### 2. Differential Evolution

**Mutation:**

$$
v_i = x_a + F \cdot (x_b - x_c)
$$

- $x_i$: aktueller Kandidat (Partikel) i
- $x_a, x_b, x_c$: drei verschiedene Kandidaten aus der Population, zufällig gewählt
- $F$: Skalierungsfaktor, steuert die Schrittweite der Mutation
- $v_i$: mutierter Kandidat, der die Differenz zwischen $x_b$ und $x_c$ in Richtung $x_a$ verschiebt

**Crossover:**

$$
u_{i,k} =
\begin{cases}
v_{i,k}, & \text{wenn } rand < CR \\
x_{i,k}, & \text{sonst}
\end{cases}
$$

- $u_i$: Trial-Vektor nach Crossover
- $k$: Index der Dimension (Feature, Radii-Komponente, …)
- $rand$: Zufallszahl in [0,1] für jede Dimension
- $CR$: Crossover-Wahrscheinlichkeit.

**Selektion:**

$$
x_i =
\begin{cases}
u_i, & f(u_i) < f(x_i)\\
x_i, & \text{sonst}
\end{cases}
$$

- $x_i$: nächster Kandidat in der Population
- $f$: Ziel- oder Kostenfunktion
- $u_i$: Trial-Kandidat vom Crossover

---

### 3. Fitnessfunktion

## Objective Function – Fitness Erklärung

Die Fitness eines Shapes wird als gewichtete Summe von Penalties und positiven Beiträgen berechnet.  
Ziel ist es, Shapes zu finden, die:

- vollständig in den Korridor passen  
- gut rotieren können  
- möglichst große Fläche haben  
- leicht unregelmäßig (nicht perfekt kreisförmig) sind  
- glatt und symmetrisch sind  
- nicht zu konkav oder extrem geformt sind  

### Mathematische Darstellung

Die Fitness $F$ eines Shapes lässt sich schreiben als:

$$
F = 
\sum_{i \in \{\text{rotation, placement, smoothness, symmetry, concavity, aspect}\}} 
- w_i \cdot p_i
+ w_\text{area} \cdot A(r)
+ w_\text{noncircular} \cdot N(r)
$$

- $p_i$ = Penalty für den Faktor $i$  
- $w_i$ = Gewicht, das die Wichtigkeit des Faktors steuert  
- $A(r)$ = Fläche des Shapes  
- $N(r)$ = Maß für die Noncircularity  

**Interpretation:**  
- Penalties werden negativ in die Fitness eingebracht → geringere Strafen = bessere Fitness  
- Fläche und Noncircularity werden positiv eingebracht → größere Shapes + leicht unregelmäßige Shapes = bessere Fitness
"""
)

# ========================================================
# TAB 3 - RESULTS - RUN DE + Animation
# ========================================================
with tabs[2]:

    st.markdown("## Differential Evolution - Simulation")

    st.sidebar.markdown("### Parameter konfigurieren")

    # Korridor Parameter
    st.sidebar.markdown("#### L-Corridor")
    corridor_width = st.sidebar.slider("corridor_width", 1.0, 3.0, 1.5)
    horizontal_length = st.sidebar.slider("horizontal_length", 4.0, 12.0, 6.0)
    vertical_length = st.sidebar.slider("vertical_length", 4.0, 12.0, 6.0)

    # DE Parameter
    st.sidebar.markdown("#### DE-Parameter")
    K = st.sidebar.slider("K (Radienanzahl)", 10, 100, 50)
    r_min = st.sidebar.slider("r_min", 0.05, 0.5, 0.1)
    r_max = st.sidebar.slider("r_max", 0.5, 2.0, 1.3)
    popsize = st.sidebar.slider("popsize", 10, 100, 32)
    F = st.sidebar.slider("F (Mutation)", 0.1, 1.5, 0.6)
    CR = st.sidebar.slider("CR (Crossover)", 0.1, 1.0, 0.8)
    generations = st.sidebar.slider("Generations", 10, 2000, 1000)
    smooth_window = st.sidebar.slider("Smoothing Window", 1, 8, 2)
    seed = st.sidebar.number_input("Seed", value=1)

    if st.button("Diff Evol starten"):

        st.info("Optimierung läuft… das kann dauern.")

        # Fortschrittsanzeige
        progress = st.progress(0)
        status = st.empty()

        def progress_callback(current_iter, total_iters, best_fitness):
            progress.progress(current_iter / total_iters)
            status.text(
                f"Iteration {current_iter}/{total_iters} - aktueller bester Score: {best_fitness:.4f}"
            )

        corridor = construct_corridor(
            corridor_width=corridor_width,
            horizontal_length=horizontal_length,
            vertical_length=vertical_length,
        )

        start_t = time.time()

        best_radii, best_poly, history = differential_evolution(
            K=K,
            r_min=r_min,
            r_max=r_max,
            popsize=popsize,
            F=F,
            CR=CR,
            generations=generations,
            smooth_window=smooth_window,
            seed=seed,
            progress_callback=progress_callback,
        )

        end_t = time.time()

        st.success(f"Fertig! Laufzeit: {end_t - start_t:.2f} Sekunden")
        st.write("### Kostenverlauf")
        st.line_chart(history)

        st.write("### Beste gefundene Form")

        # Bewegung simulieren
        placed = place_polygon_against_corridor(best_poly, corridor)
        feasible, max_rot, path = move_and_rotate_smooth(corridor, placed)

        if len(path) == 0:
            st.warning("Diese Form konnte nicht durch den Korridor bewegt werden.")
        else:
            st.success("Eine gültige Bewegung wurde gefunden!")

        # Animation erzeugen (GIF)
        with st.spinner("Erzeuge Animation..."):

            fig, ax = plt.subplots()
            if len(path) == 0:
                st.error("Keine Animation verfügbar - Pfad ist leer. Die Form konnte sich nicht bewegen.")
            else:
                frames = []
                for poly in path:
                    fig, ax = plt.subplots()
                    ax.set_aspect("equal")
                    ax.set_axis_off()

                    cx, cy = corridor.exterior.xy
                    ax.fill(cx, cy, color="lightgray", alpha=0.5)

                    px, py = poly.exterior.xy
                    ax.fill(px, py, color="tomato", alpha=0.8)

                    # Frame erzeugen
                    fig.canvas.draw()
                    img = np.array(fig.canvas.renderer.buffer_rgba())
                    frames.append(img)
                    plt.close(fig)

                gif_path = "sofa_animation.gif"
                imageio.mimsave(gif_path, frames, fps=10)

                st.image(gif_path, caption="Optimierte Bewegung")



# ========================================================
# TAB 4 - DISCUSSION
# ========================================================
with tabs[3]:

    st.markdown("## Diskussion")

    st.markdown(
        r"""
        ### Auswirkungen der Parameter

        **Populationsgröße**
        - große Population → mehr Diversität  
        - bessere Ergebnisse, aber langsamer  

        **Mutationsfaktor $F$**
        - niedrig: lokale Suche  
        - hoch: aggressive Exploration, instabil  

        **Crossover-Rate $CR$**
        - hoch: schnelle Veränderung  
        - niedrig: konservative Suche  

        **Korridorbreite**
        - je breiter der Korridor, desto größer können Formen werden  
        - schmale Korridore → extrem schwer rotierbare Formen  

        **Radienanzahl $K$**
        - klein: glatte Formen  
        - groß: komplexe konkave Strukturen möglich  

        ### Fazit
        
        Das Moving-Sofa-Problem ist stark nichtkonvex und hochgradig schwierig.  
        Selbst moderne Evolutionsalgorithmen nähern sich nur *lokalen Optima*.  
        
        """
    )
