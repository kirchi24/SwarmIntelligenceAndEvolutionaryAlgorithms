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
        x = [x_1, x_2, \\dots, x_{54}]
        $$

        Wir verwenden **kontinuierliche Werte** für die Optimierung:
        - PSO arbeitet besser in einem kontinuierlichen Raum als in einem binären Raum.
        - Jeder Wert $x_i \\in \\mathbb{R}$ beschreibt die "Stärke", mit der Feature $i$ ausgewählt wird.
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
        best_features, best_fitness, best_f1, fitness_curve, f1_curve = results

        st.info(
            f"PSO-Durchlaufzeit: {elapsed_time:.2f} Sekunden \t Zeit pro Forest Auswertung: {elapsed_time * 1000 / (n_particles * iterations)} Millisekunden"
        )
        st.success(
            f"PSO abgeschlossen! Bester Fitnesswert: {best_fitness:.4f} mit einem F1-Score von {best_f1:.4f} \t Anzahl ausgewählter Features: {sum(best_features)} / 54"
        )

        fig = go.Figure()

        # Fitnesskurve
        fig.add_trace(
            go.Scatter(
                y=fitness_curve,
                mode="lines+markers",
                name="Fitness",
                line=dict(color="green", width=3)
            )
        )

        # F1-Score-Kurve
        fig.add_trace(
            go.Scatter(
                y=f1_curve,
                mode="lines+markers",
                name="F1-Score",
                line=dict(color="blue", width=3, dash="dash")
            )
        )

        fig.update_layout(
            title="Fitness und F1-Score Verlauf über die Iterationen",
            xaxis=dict(title="Iteration"),
            yaxis=dict(
                title="Wert",
                range=[0, 1],  # Achse bei 0 starten
            ),
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # best_features: 0/1-Maske
        feature_names = [f"f{i:02d}" for i in range(len(best_features))]
        df_mask = pd.DataFrame({
            "Feature": feature_names,
            "Selected": best_features
        })

        # Nur ausgewählte Features auflisten
        selected_features = df_mask[df_mask["Selected"] == 1]["Feature"].tolist()
        st.write(f"### Ausgewählte Features ({len(selected_features)})")
        st.write(selected_features)

# =====================================================================
# TAB 4 — DISCUSSION
# =====================================================================
with tabs[3]:
    st.markdown("## Diskussion")

    st.markdown(
        """
        ### Analyse der Ergebnisse

        Die Ergebnisse aus Tab 3 zeigen, dass PSO in der Lage ist, aus den 54 ursprünglichen Features ein deutlich kleineres, aber trotzdem leistungsfähiges Subset auszuwählen. In unserem Lauf wurden **20 Features** ausgewählt, was einer Reduktion von rund **63 %** entspricht.  

        Der daraus resultierende **F1-Score** und eine **Fitness** entsprechen in etwa dem, was bei einer datensatzunabhängigen, heuristischen Feature-Selection mit PSO zu erwarten ist. Der F1-Score steigt anfangs sichtbar an, stabilisiert sich aber relativ früh, was darauf hindeutet, dass der Schwarm eine geeignete Region im Suchraum gefunden hat und danach eher Feintuning stattfindet.

        ### Erwartete vs. unerwartete Beobachtungen

        **Erwartet:**
        - Die Fitness verbessert sich hauptsächlich in den ersten Iterationen und flacht danach ab.
        - PSO bevorzugt kleinere Feature-Subsets, da die Fitness eine Sparsity-Komponente enthält.
        - Die Varianz im F1-Score bleibt moderat, da ExtraTrees relativ stabil ist.

        **Unerwartet:**
        - Die Verbesserung des F1-Scores fällt geringer aus als die Verbesserung der Gesamtfitness.  
        → Das deutet darauf hin, dass PSO stärker von der Sparsity-Komponente profitiert als von der Modellperformance.
        - Trotz 20 Iterationen gibt es nur kleine Sprünge im globalen F1-Score.  
        → Vermutlich liegt dies daran, dass das Klassifikationsproblem (Covertype) sehr komplex und stark nichtlinear ist.

        ### Effizienz und Rechenaufwand

        Der größte Engpass des PSO-Ansatzes ist eindeutig die **Rechenzeit**.  
        In jeder Iteration müssen alle Partikel **einen kompletten ExtraTrees-Classifier trainieren** – und das bei einem Datensatz mit **581.000 Beobachtungen**.

        Dies führt zu:

        - **Lange Gesamtlaufzeit** für n Partikel × i Iterationen  
        - **~500 ms pro Klassifikator-Training**, trotz ExtraTrees (der bereits zu den schnelleren Baummodellen gehört)
        - Einem quadratischen Wachstumsverhalten:  
        mehr Partikel × mehr Iterationen × viele Beobachtungen = exponentiell steigender Aufwand

        Das erklärt auch, warum PSO in der Literatur häufig *nicht* auf großen Tabellendaten eingesetzt wird, sondern eher in Bereichen wie Hyperparameter-Optimierung oder Feature-Selection für kleinere Datensätze.

        ### Qualität der Lösung

        Die finale Lösung (20 Features, F1-Score ~0.32) ist angesichts der Datensatzgröße, der Nichtlinearität und der Komplexität des Problems durchaus solide.  
        Sie zeigt:

        - PSO findet **kompaktere Feature-Subsets**, ohne zu viel predictive power zu verlieren.
        - Die Mischung aus F1 und Sparsity führt zu einem sinnvollen Trade-off.
        - Der Algorithmus konvergiert stabil.

        Allerdings liegt die Performance etwas unter dem, was ein voll trainiertes, komplexeres Modell mit allen Features erreichen könnte.

        ### Limitierungen des Ansatzes

        1. **Hoher Rechenaufwand**  
        Jede PSO-Iteration erfordert ein erneutes Training des Klassifikators → O(Particles × Iterations × TrainingTime).

        2. **Hohe Dimensionalität + große Datenmenge**  
        Der Covertype-Datensatz ist für Black-Box-Optimierer sehr anspruchsvoll.

        3. **Keine Qualitätssicherung bei lokaler Konvergenz**  
        PSO bietet keine Garantie, dass die Lösung global optimal ist.

        4. **Sigmoid-Binarisierung kann rauschig sein**  
        Kleine Änderungen in den Positionswerten können Features ein/aus schalten.

        ### Verbesserungsmöglichkeiten

        1. **GPU-Beschleunigung**  
        - Einsatz von cuML ExtraTrees → 10–20× schnellere Trainingszeit  
        - Reduziert PSO-Laufzeiten dramatisch

        2. **Surrogate Modeling**  
        - Statt den Klassifikator jedes Mal neu zu trainieren  
        - Nutzung eines LightGBM-Surrogates oder Approximationsmodells für die Fitness  
        - Danach Feinoptimierung nur auf guten Kandidaten

        3. **Downsampling / Mini-Batch-Evaluation**  
        - Klassifikator nur auf einem Teil des Datensatzes trainieren, um Fitness schnell zu approximieren

        4. **Binarer PSO oder hybride Algorithmen**  
        - BPSO, GA-PSO-Mix, oder Sparse-PSO können bei Feature Selection bessere Ergebnisse liefern

        5. **Adaptive Parametersteuerung**  
        - w, c1, c2 können iterativ angepasst werden (z.B. lineare Reduktion von w)

        6. **Feature Pre-Ranking vor PSO**  
        - Z. B. über ExtraTrees-Importances  
        - PSO muss dann nicht bei 54 Features starten, sondern nur bei den Top-30  

        ---
        """
    )
