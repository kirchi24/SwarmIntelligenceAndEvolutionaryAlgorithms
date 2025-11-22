import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

from src.ParticleSwarm.pso_algorithm import (
    run_pso,
)

# Streamlit-Konfiguration
st.set_page_config(
    page_title="Particle Swarm Optimization (PSO) - Feature Selection",
    layout="wide",
)

st.title("Particle Swarm Optimization (PSO) für Feature Selection - Covertype Dataset")

# Tabs
tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])

# =====================================================================
# TAB 1 — INTRODUCTION
# =====================================================================
with tabs[0]:
    st.markdown("## Einführung in die PSO-basierte Feature Selection")

    st.markdown(
        """
        **Particle Swarm Optimization (PSO)** ist ein Schwarm-basierter Optimierungsalgorithmus, 
        der vom sozialen Verhalten von Schwärmen (Vögel, Fische, Insekten) inspiriert ist.

        In diesem Projekt wird PSO verwendet, um **Features aus dem Covertype-Datensatz** auszuwählen,
        sodass ein Klassifikationsmodell bessere Vorhersagen trifft und gleichzeitig **weniger Features** nutzt.

        ---
        ### Warum Feature Selection?
        - Reduktion der Dimensionalität
        - Kürzere Trainingszeit
        - Weniger Overfitting
        - Höhere Modellinterpretierbarkeit

        ---
        ### Warum PSO für Feature Selection?
        - PSO benötigt keine Gradienten
        - Gute Balance zwischen Exploration und Exploitation
        - Kann diskrete Entscheidungen (Feature an/aus) durch Sigmoid-Binarisierung treffen
        ---
        ### Optimierungsziel

        Die Fitnessfunktion lautet:
        """
    )

    st.latex(
        r"""
        \text{fitness}(x) = 
        \alpha \cdot \text{F1}_\text{macro}(x)
        + (1-\alpha)\left(1 - \frac{n_\text{selected}}{N_\text{features}}\right)
        """
    )

    st.markdown(
        """
        wobei  
        - $ \\text{F1}_\\text{macro} $ = mittlere Klassen-F1-Score  
        - $ n_\text{selected} $ = Anzahl ausgewählter Features  
        - $ N_\text{features} $ = Gesamtzahl der Features  
        """
    )


# =====================================================================
# TAB 2 — METHODS
# =====================================================================
with tabs[1]:
    st.markdown("## Methoden - PSO für Feature Selection")
    st.markdown(
        """
        Jeder Partikel kodiert ein **Feature-Subset** als Vektor aus Realszahlen:

        $$
        x = [x_1, x_2, \dots, x_{54}]
        $$

        Wir verwenden **kontinuierliche Werte** für die Optimierung:
        - PSO arbeitet besser in einem kontinuierlichen Raum als in einem binären Raum.
        - Jeder Wert $x_i \in \mathbb{R}$ beschreibt die "Stärke", mit der Feature $i$ ausgewählt wird.
        - Durch die Sigmoid-Funktion wird daraus eine binäre Maske für die Feature-Auswahl.
        """
    )
    st.latex(
        r"""
        S(x_i) = \frac{1}{1 + e^{-x_i}}, \qquad
        \hat{x}_i =
        \begin{cases}
        1, & S(x_i) > 0.5 \\
        0, & \text{otherwise}
        \end{cases}
        """
    )

    # -----------------------------
    # 2. PSO-Update-Regeln
    # -----------------------------
    st.subheader("2. PSO-Update-Regeln")
    st.latex(
        r"""
        v_i \leftarrow \omega v_i 
        + c_1 r_1 (p_{\text{best}_i} - x_i)
        + c_2 r_2 (g_{\text{best}} - x_i)
        """
    )
    st.latex(r"x_i \leftarrow x_i + v_i")
    st.markdown(
        """
        - **$\\omega$** = Trägheitsgewicht (wie stark die aktuelle Bewegung beibehalten wird)  
        - **$c_1$** = kognitiver Faktor (eigene Erfahrung)  
        - **$c_2$** = sozialer Faktor (beste Partikel im Schwarm)  
        - **$r_1, r_2$** = Zufallszahlen (0 bis 1)
        """
    )

    # -----------------------------
    # 3. Fitnessfunktion
    # -----------------------------
    st.subheader("3. Fitnessfunktion")

    st.markdown(
        """
        Die Fitness bewertet ein Feature-Subset anhand von zwei Kriterien:
        1. **Vorhersagequalität**: Makro-F1-Score des Klassifikators  
        2. **Sparsity**: möglichst wenige Features auswählen  

        Für die Vorhersage verwenden wir **ExtraTreesClassifier**:
        - Schneller als RandomForest, da:
            - Keine Bootstraps notwendig
            - Zufällige Splits werden direkt gewählt
        - Liefert stabile Feature-Importances und gute Approximation für die Fitness in PSO
        """
    )
    st.latex(
        r"""
        \text{fitness}(x) = 
        \alpha \cdot \text{F1}_\text{macro}(x)
        + (1-\alpha)\left(1 - \frac{n_\text{selected}}{N_\text{features}}\right)
        """
    )


    # Plot: Beispielhafte Visualisierung von Velocity Updates
    st.subheader(
        "Beispiel: Einfluss von $\\omega, c_1, c_2$ auf die Bewegung eines Partikels"
    )

    iterations = np.arange(30)
    inertia = np.exp(-0.15 * iterations)
    cognitive = np.sin(0.4 * iterations) * 0.5 + 0.5
    social = np.cos(0.2 * iterations) * 0.5 + 0.5

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=iterations, y=inertia, mode="lines", name="Inertia weight")
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=cognitive, mode="lines", name="Cognitive term")
    )
    fig.add_trace(go.Scatter(x=iterations, y=social, mode="lines", name="Social term"))
    fig.update_layout(
        title="PSO-Dynamik der Update-Komponenten",
        xaxis_title="Iteration",
        yaxis_title="Stärke",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 3 — RESULTS (PSO ausführen)
# =====================================================================
with tabs[2]:
    st.markdown("## PSO - Simulation & Ergebnisse")

    # Sidebar Parameters
    st.sidebar.markdown("### PSO Parameter")

    n_particles = st.sidebar.slider("Anzahl Partikel", 5, 50, 20)
    iterations = st.sidebar.slider("Iterationen", 5, 100, 20)
    w = st.sidebar.slider("Inertia ($\\omega$)", 0.1, 1.0, 0.7)
    c1 = st.sidebar.slider("$c_1$ (kognitiv)", 0.0, 3.0, 1.5)
    c2 = st.sidebar.slider("$c_2$ (sozial)", 0.0, 3.0, 1.5)
    alpha = st.sidebar.slider("Alpha (Accuracy-Gewicht)", 0.3, 1.0, 0.7)

    st.sidebar.markdown("### RandomForest Parameter")
    n_estimators = st.sidebar.slider("n_estimators", 2, 25, 5)
    max_depth = st.sidebar.slider("max_depth", 2, 25, 5)

    if st.button("PSO starten"):
        st.info("PSO wird ausgeführt… das kann bei großen Datensätzen dauern.")

        # Fortschrittsanzeige
        progress = st.progress(0)
        status = st.empty()

        def progress_callback(current_iter, total_iters, best_fitness):
            progress.progress(current_iter / total_iters)
            status.text(
                f"Iteration {current_iter}/{total_iters} - aktueller bester Score: {best_fitness:.4f}"
            )

        start_time = time.time()
        # Algorithmus starten
        results = run_pso(
            n_particles=n_particles,
            iterations=iterations,
            w=w,
            c1=c1,
            c2=c2,
            alpha=alpha,
            n_estimators=n_estimators,
            max_depth=max_depth,
            progress_callback=progress_callback,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.info(f"PSO-Durchlaufzeit: {elapsed_time:.2f} Sekunden")
        st.info(
            f"Zeit pro Forest Auswertung: {elapsed_time * 1000 / (n_particles * iterations)} Millisekunden"
        )

        best_features, best_fitness, fitness_curve = results

        st.success(f"PSO abgeschlossen! Bester Fitnesswert: {best_fitness:.4f}")
        st.markdown(f"**Anzahl ausgewählter Features:** {sum(best_features)} / 54")

        # Plot: Fitnesskurve
        fig_curve = go.Figure()
        fig_curve.add_trace(
            go.Scatter(
                y=fitness_curve,
                mode="lines+markers",
                line=dict(color="green", width=3),
            )
        )
        fig_curve.update_layout(
            title="Fitnessverlauf über die Iterationen",
            xaxis_title="Iteration",
            yaxis_title="Fitness",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Feature-Maske anzeigen
        st.write("### Ausgewählte Features (1 = ausgewählt)")
        df_mask = pd.DataFrame(
            {"Feature": [f"f{i:02d}" for i in range(54)], "Selected": best_features}
        )
        st.dataframe(df_mask)

# =====================================================================
# TAB 4 — DISCUSSION
# =====================================================================
with tabs[3]:
    st.markdown("## Diskussion")

    st.markdown(
        """
        ### Erkenntnisse aus der PSO-Feature-Selection

        **Vorteile:**
        - PSO findet auch bei 54 Dimensionen robuste Subsets  
        - Gute Balance zwischen Genauigkeit und Feature-Reduktion  
        - Flexibel an Klassifikator (RF) anbinden  

        **Nachteile:**
        - Rechenintensiv (RF-Training pro Partikel)  
        - Häufig viele Iterationen notwendig  
        - Keine Garantie für globales Optimum  

        ### Verbesserungspotenzial
        - Parallelisierung über GPU-RandomForest (cuML)  
        - Hyperparameter-Tuning für ($\\omega, c_1, c_2$)  
        - Kombination mit genetischen Algorithmen  
        - Modifizierte Binary-PSO-Varianten  
        """
    )
