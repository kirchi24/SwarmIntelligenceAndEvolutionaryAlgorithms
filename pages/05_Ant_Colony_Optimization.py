import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.AntColonyOptimization.aco_algorithm import run_aco

# Streamlit-Konfiguration
st.set_page_config(
    page_title="Ant Colony Optimization (ACO) - Nurse Scheduling",
    layout="wide",
)

st.title("Ant Colony Optimization (ACO) für das Nurse Scheduling Problem")

# Tabs erstellen
tabs = st.tabs(["Introduction", "Methods", "Results", "Discussion"])

# ---------------------------
# Tab 1: Introduction
# ---------------------------
with tabs[0]:
    st.markdown("## Einführung")
    st.markdown(
        """
    Ant Colony Optimization (ACO) ist eine metaheuristische Methode, inspiriert vom Nahrungssucheverhalten realer Ameisen. 
    Sie eignet sich besonders für kombinatorische Optimierungsprobleme, z. B. das Travelling Salesman Problem (TSP) 
    oder das Nurse Scheduling Problem (NSP).

    **Stärken:**
    - Kann komplexe, multi-constraint Probleme lösen  
    - Flexibel bei harten und weichen Einschränkungen  

    **Schwächen:**
    - Parameterwahl (ρ, Q, Anzahl Ameisen) kann die Lösung stark beeinflussen  
    - Berechnungsintensiv bei großen Problemgrößen
    """
    )

# ---------------------------
# Tab 2: Methods
# ---------------------------
with tabs[1]:
    st.markdown("## Methoden / Implementierung")
    st.markdown("### Pheromon-Aktualisierung")
    st.subheader("Wie funktionieren Pheromone in der Ameisenkolonieoptimierung?")

    st.markdown(
        """
    In der **Ameisenkolonieoptimierung (Ant Colony Optimization, ACO)** wird die Suche nach einer guten Lösung durch **Pheromone** gesteuert.  
    Jede Ameise hinterlässt auf den Pfaden (oder hier: *Dienstzuweisungen*) eine **Pheromonspur**.  
    Diese Spur ist umso stärker, je **besser** die gefundene Lösung ist.  

    ---
    """
    )
    st.latex(
        r"\tau_{n,d,s} \leftarrow (1 - \rho)\,\tau_{n,d,s} + \sum_k \Delta\tau_{n,d,s}^{(k)}"
    )

    st.markdown("Dabei gilt:")
    st.latex(
        r"""
    \Delta\tau_{n,d,s}^{(k)} =
    \begin{cases}
    \dfrac{Q}{1 + L_k}, & \text{wenn Ameise k die Zuweisung gesetzt hat} \\
    0, & \text{sonst}
    \end{cases}
    """
    )

    st.markdown("**Parameter:**")
    st.markdown(
        """
    - ρ — Verdunstungsrate (z. B. 0,1 → 10 % Verlust pro Iteration)  
    - Q — Verstärkungsfaktor für abgelegtes Pheromon  
    - Lₖ — Kosten / Score der Ameise k (je kleiner, desto besser)
    """
    )

    st.markdown("**Intuition:**")
    st.markdown(
        """
    - Schlechte Lösungen verlieren Gewicht: $ \\tau \\leftarrow (1-\\rho)\, \\tau $
    - Gute Lösungen verstärken Pfade: $\\tau \leftarrow \\tau + \\dfrac{Q}{1 + L_k}$  
    - Gleichgewicht zwischen Vergessen und Lernen fokussiert den Algorithmus auf gute Lösungen
    """
    )

    # Beispielhafte Simulation der Pheromonentwicklung
    iterations = np.arange(0, 50)
    pheromone_good = 1 - np.exp(
        -0.1 * iterations
    )  # simulierte Verstärkung bei guter Lösung
    pheromone_bad = 0.3 * np.exp(
        -0.05 * iterations
    )  # simulierte Verdunstung schlechter Lösung

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=pheromone_good,
            mode="lines+markers",
            name="Gute Lösung",
            line=dict(color="green", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=pheromone_bad,
            mode="lines+markers",
            name="Schlechte Lösung",
            line=dict(color="gray", dash="dot", width=2),
        )
    )

    fig.update_layout(
        title="Entwicklung der Pheromonintensität über Iterationen",
        xaxis_title="Iteration",
        yaxis_title="Pheromonwert τ",
        width=800,
        height=400,
        legend_title="Lösungsqualität",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        """
    **Interpretation:**
    - Die *grüne Kurve* zeigt, wie sich das Pheromon auf einem guten Pfad mit der Zeit verstärkt.  
    - Die *graue Kurve* zeigt, wie schwache Pheromone auf schlechten Pfaden verdunsten.  
    - Im Gleichgewicht konzentrieren sich die Ameisen auf wenige, qualitativ hochwertige Lösungen.
    """
    )

# ---------------------------
# Tab 3: Results
# ---------------------------
with tabs[2]:
    st.markdown("## Ergebnisse und Simulation")

    # ---------------------------
    # Interaktive Parameter
    # ---------------------------
    N = st.slider("Anzahl Krankenschwestern", 3, 20, 10, 1)
    D = st.slider("Anzahl Tage", 1, 14, 7, 1)
    S = st.slider("Anzahl Schichten pro Tag", 1, 5, 3, 1)

    alpha = st.slider("Alpha (Pheromone Gewicht)", 0.1, 5.0, 1.0, 0.1)
    beta = st.slider("Beta (Heuristik Gewicht)", 0.1, 10.0, 5.0, 0.1)
    rho = st.slider("Verdunstungsrate ρ", 0.0, 1.0, 0.1, 0.01)
    Q = st.slider("Pheromon Verstärkung Q", 1.0, 200.0, 100.0, 1.0)
    num_ants = st.slider("Anzahl Ameisen pro Iteration", 1, 100, 30, 1)
    max_iters = st.slider("Maximale Iterationen", 1, 500, 50, 1)

    if st.button("ACO ausführen"):
        tau_init = np.ones((N, D, S)) * 0.1

        # Fortschrittsanzeige (kein Spinner!)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Callback für Fortschritt
        def update_progress(current, total, best_score):
            progress_bar.progress(current / total)
            progress_text.text(
                f"Iteration {current}/{total} - Aktueller bester Score: {best_score:.2f}"
            )

        # --- ACO ausführen ---
        (
            best_schedule,
            best_score,
            breakdown,
            tau_final,
            tau_history,
            best_score_history,
        ) = run_aco(
            tau_init,
            num_ants=num_ants,
            max_iters=max_iters,
            alpha=alpha,
            beta=beta,
            rho=rho,
            Q=Q,
            verbose=False,
            seed=42,
            return_tau_history=True,
            progress_callback=update_progress,
        )

        # Fortschrittsanzeige abschließen
        progress_bar.empty()
        progress_text.empty()

        st.success(f"ACO abgeschlossen! Beste Punktzahl: {best_score:.2f}")

        # ---------------------------
        # Scoreverlauf Plotly
        # ---------------------------

        fig_score = go.Figure()
        fig_score.add_trace(
            go.Scatter(
                y=best_score_history,
                mode="lines+markers",
                name="Best Score",
                line=dict(color="green", width=3),
            )
        )
        fig_score.update_layout(
            title="Entwicklung des besten Scores über die Iterationen",
            xaxis_title="Iteration",
            yaxis_title="Bester Score",
            template="plotly_white",
            width=800,
            height=400,
        )
        st.plotly_chart(fig_score, use_container_width=True)

        # ---------------------------
        # Heatmap der besten Schedule
        # ---------------------------
        st.write("### Beste Schedule pro Schicht (1 = Dienst, 0 = frei)")
        st.write("FS = Frühschicht, SS = Spätschicht, NS = Nachtschicht")

        best_schedule_num = np.nan_to_num(best_schedule, 0)
        columns = [
            f"Tag {d+1}_{label}" for d in range(D) for label in ["FS", "SS", "NS"][:S]
        ]
        data_2d = best_schedule_num.reshape(N, D * S)

        colorscale = [[0, "white"], [1, "lightgreen"]]
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=data_2d,
                x=columns,
                y=[f"KS{i+1}" for i in range(N)],
                colorscale=colorscale,
                showscale=False,
                hoverongaps=False,
                text=data_2d,
                texttemplate="%{text}",
            )
        )

        shapes = []
        for d in range(D):
            x0 = d * S - 0.5
            x1 = (d + 1) * S - 0.5
            if d % 2 == 0:
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=x0,
                        x1=x1,
                        y0=0,
                        y1=1,
                        fillcolor="rgba(230,230,230,0.6)",
                        line_width=0,
                        layer="below",
                    )
                )

        fig_heat.update_layout(
            title="Dienstplan Heatmap (grün = Schicht zugewiesen)",
            xaxis_nticks=21,
            yaxis_autorange="reversed",
            width=1000,
            height=400,
            shapes=shapes,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ---------------------------
        # Barplot: Gesamtanzahl der Schichten
        # ---------------------------
        st.write("### Gesamtanzahl der Schichten pro Krankenschwester")

        shifts_per_nurse = np.nansum(best_schedule, axis=(1, 2))

        df_shifts = pd.DataFrame(
            {
                "Krankenschwester": [f"KS{i+1}" for i in range(N)],
                "Anzahl Schichten": shifts_per_nurse,
            }
        )

        # Farbe manuell festlegen nach Regel:
        def color_rule(x):
            if x < 4:
                return "red"
            elif x == 4:
                return "green"
            else:
                return "gray"

        df_shifts["Farbe"] = df_shifts["Anzahl Schichten"].apply(color_rule)

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=df_shifts["Krankenschwester"],
                y=df_shifts["Anzahl Schichten"],
                text=df_shifts["Anzahl Schichten"],
                textposition="outside",
                marker_color=df_shifts["Farbe"],
            )
        )
        fig_bar.update_layout(
            title="Gesamtanzahl der Schichten pro Krankenschwester",
            yaxis_title="Anzahl Schichten",
            yaxis_range=[0, 7],
            xaxis_title="Krankenschwester",
            yaxis=dict(tick0=0, dtick=1),
            width=800,
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ---------------------------
        # 3D-Pheromon-Animation mit besserer Farbabstufung
        # ---------------------------
        st.write("### Entwicklung der Pheromon-Intensität über die Iterationen")

        x, y, z, c, itr_list = [], [], [], [], []
        for itr, tau_m in enumerate(tau_history):
            for n_ in range(N):
                for d_ in range(D):
                    for s_ in range(S):
                        x.append(n_)
                        y.append(d_)
                        z.append(s_)
                        c.append(tau_m[n_, d_, s_])
                        itr_list.append(itr + 1)

        df = pd.DataFrame(
            {"Nurse": x, "Day": y, "Shift": z, "Pheromone": c, "Iteration": itr_list}
        )

        fig_tau = px.scatter_3d(
            df,
            x="Nurse",
            y="Day",
            z="Shift",
            color="Pheromone",
            animation_frame="Iteration",
            color_continuous_scale="Viridis_r",
            range_color=[df["Pheromone"].min(), df["Pheromone"].max()],
            opacity=0.8,
        )

        fig_tau.update_layout(
            scene=dict(
                xaxis_title="Krankenschwester",
                yaxis_title="Tag",
                zaxis_title="Schicht",
            ),
            title="Pheromonentwicklung über Iterationen",
            width=900,
            height=600,
            template="plotly_white",
        )

        st.plotly_chart(fig_tau, use_container_width=True)

# ---------------------------
# Tab 4: Discussion
# ---------------------------
with tabs[3]:
    st.markdown("## Diskussion")
    st.markdown(
        """
        **Vorteile der ACO-Methode:**
        - Flexibel bei der Modellierung komplexer Einschränkungen  
        - Gute Balance zwischen Exploration und Exploitation durch Pheromonmechanismus  

        **Nachteile und Herausforderungen:**
        - Sensitivität gegenüber Parameterwahl (ρ, Q, Anzahl Ameisen)  
        - Rechenintensiv bei großen Problemgrößen  
        - Gefahr des Premature Convergence bei zu starker Fokussierung auf bestimmte Pfade  

        **Mögliche Verbesserungen:**
        - Adaptive Parameteranpassung während der Laufzeit  
        - Hybridisierung mit anderen Optimierungsverfahren (z. B. genetische Algorithmen)  
        - Erweiterte Heuristiken zur Initialisierung und Pfadauswahl  
        """
    )
