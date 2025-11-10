import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

    st.markdown("""
    In der **Ameisenkolonieoptimierung (Ant Colony Optimization, ACO)** wird die Suche nach einer guten Lösung durch **Pheromone** gesteuert.  
    Jede Ameise hinterlässt auf den Pfaden (oder hier: *Dienstzuweisungen*) eine **Pheromonspur**.  
    Diese Spur ist umso stärker, je **besser** die gefundene Lösung ist.  

    ---
    """
    )
    st.latex(r"\tau_{n,d,s} \leftarrow (1 - \rho)\,\tau_{n,d,s} + \sum_k \Delta\tau_{n,d,s}^{(k)}")

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
    - Schlechte Lösungen verlieren Gewicht: $ \\tau \leftarrow (1-\\rho)\, \\tau $
    - Gute Lösungen verstärken Pfade: $\\tau \leftarrow \\tau + \dfrac{Q}{1 + L_k}$  
    - Gleichgewicht zwischen Vergessen und Lernen fokussiert den Algorithmus auf gute Lösungen
    """
    )

 
    # Beispielhafte Simulation der Pheromonentwicklung
    iterations = np.arange(0, 50)
    pheromone_good = 1 - np.exp(-0.1 * iterations)  # simulierte Verstärkung bei guter Lösung
    pheromone_bad = 0.3 * np.exp(-0.05 * iterations)  # simulierte Verdunstung schlechter Lösung

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations,
        y=pheromone_good,
        mode="lines+markers",
        name="Gute Lösung",
        line=dict(color="green", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=iterations,
        y=pheromone_bad,
        mode="lines+markers",
        name="Schlechte Lösung",
        line=dict(color="gray", dash="dot", width=2)
    ))

    fig.update_layout(
        title="Entwicklung der Pheromonintensität über Iterationen",
        xaxis_title="Iteration",
        yaxis_title="Pheromonwert τ",
        width=800,
        height=400,
        legend_title="Lösungsqualität",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Interpretation:**
    - Die *grüne Kurve* zeigt, wie sich das Pheromon auf einem guten Pfad mit der Zeit verstärkt.  
    - Die *graue Kurve* zeigt, wie schwache Pheromone auf schlechten Pfaden verdunsten.  
    - Im Gleichgewicht konzentrieren sich die Ameisen auf wenige, qualitativ hochwertige Lösungen.
    """)

# ---------------------------
# Tab 3: Results
# ---------------------------
with tabs[2]:
    st.markdown("## Ergebnisse und Simulation")

    # ---------------------------
    # Interaktive Parameter
    # ---------------------------
    N = st.number_input("Anzahl Krankenschwestern", min_value=3, max_value=20, value=10, step=1)
    D = st.number_input("Anzahl Tage", min_value=1, max_value=14, value=7, step=1)
    S = st.number_input("Anzahl Schichten pro Tag", min_value=1, max_value=5, value=3, step=1)

    alpha = st.slider("Alpha (Pheromone Gewicht)", 0.1, 5.0, 1.0, 0.1)
    beta = st.slider("Beta (Heuristik Gewicht)", 0.1, 10.0, 5.0, 0.1)
    rho = st.slider("Verdunstungsrate ρ", 0.0, 1.0, 0.1, 0.01)
    Q = st.slider("Pheromon Verstärkung Q", 1.0, 200.0, 100.0, 1.0)
    num_ants = st.number_input("Anzahl Ameisen pro Iteration", min_value=1, max_value=100, value=30, step=1)
    max_iters = st.number_input("Maximale Iterationen", min_value=1, max_value=500, value=50, step=1)

    if st.button("ACO ausführen"):
        # Initialize pheromone
        tau_init = np.ones((N,D,S)) * 0.1

        # ---------------------------
        # Heuristikfunktion
        # ---------------------------
        def eta_function(schedule, n, d, s):
            schedule[n,d,s] = 1
            score, _ = heuristic_score(schedule)
            schedule[n,d,s] = np.nan
            return 1 / (1 + score)

        def construct_schedule(tau, alpha, beta):
            schedule = np.full((N,D,S), np.nan)
            for d_ in range(D):
                for s_ in range(S):
                    etas = np.array([eta_function(schedule,n,d_,s_) for n in range(N)])
                    probs = (tau[:,d_,s_]**alpha)*(etas**beta)
                    probs /= probs.sum()
                    chosen = np.random.choice(N, size=min(2,N), replace=False, p=probs)
                    schedule[chosen,d_,s_] = 1
            return schedule

        def heuristic_score(schedule, required_per_shift=2, penalties=[10000,5000,3000,400,300]):
            morning, night = 0, S-1
            coverage_deficit = int(np.nansum(np.maximum(0, required_per_shift - np.nansum(schedule,axis=0))))
            rest_violations = int(np.nansum(np.logical_and(
                np.nan_to_num(schedule[:,:-1,night]), np.nan_to_num(schedule[:,1:,morning])
            ))) if D>1 else 0
            daily_over = float(np.nansum(np.maximum(0,np.nansum(schedule,axis=2)-2)))
            shifts_per_nurse = np.nansum(schedule,axis=(1,2))
            mean_shifts = float(np.nanmean(shifts_per_nurse))
            fairness_penalty = float(np.nansum((shifts_per_nurse-mean_shifts)**2))
            days_off = np.nansum(np.nansum(schedule,axis=2)==0,axis=1)
            dayoff_violations = int(np.nansum(days_off==0))
            hard = penalties[0]*coverage_deficit + penalties[1]*rest_violations + penalties[2]*daily_over
            soft = penalties[3]*fairness_penalty + penalties[4]*dayoff_violations
            return hard+soft, {}

        def update_pheromones(tau, all_schedules, scores, rho, Q):
            tau *= (1-rho)
            for s, sc in zip(all_schedules,scores):
                mask = ~np.isnan(s) & (s==1)
                tau[mask] += Q/(1+sc)
            return tau

        # ---------------------------
        # ACO loop
        # ---------------------------
        tau = tau_init.copy()
        best_score = float("inf")
        best_schedule = None
        tau_history = []

        progress_bar = st.progress(0)
        for itr in range(max_iters):
            all_schedules = [construct_schedule(tau, alpha, beta) for _ in range(num_ants)]
            scores = [heuristic_score(s)[0] for s in all_schedules]
            tau = update_pheromones(tau, all_schedules, scores, rho, Q)
            tau_history.append(tau.copy())
            local_best_idx = np.argmin(scores)
            if scores[local_best_idx] < best_score:
                best_score = scores[local_best_idx]
                best_schedule = all_schedules[local_best_idx]
            progress_bar.progress((itr+1)/max_iters)

        st.success(f"ACO abgeschlossen! Beste Punktzahl: {best_score:.2f}")

        # ---------------------------
        # Visualisierung: Heatmaps pro Schicht
        # ---------------------------
        st.write("### Beste Schedule pro Schicht (1 = Nurse arbeitet, 0 = frei)")
        st.write(" FS = Frühschicht, SS = Spätschicht, NS = Nachtschicht ")

        best_schedule_num = np.nan_to_num(best_schedule, 0)

        # Spaltenlabels: Tag1_FS, Tag1_SS, Tag1_NS, Tag2_FS ...
        columns = []
        for d in range(D):
            for s_label in ["FS", "SS", "NS"]:
                columns.append(f"Tag {d+1}_{s_label}")

        # 2D-Daten: Nurses × (Days*Shifts)
        data_2d = best_schedule_num.reshape(N, D * S)

        # Farbskala
        colorscale = [[0, 'white'], [1, 'lightgreen']]

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data_2d,
            x=columns,
            y=[f"KS{i+1}" for i in range(N)],
            colorscale=colorscale,
            showscale=False,
            hoverongaps=False,
            text=data_2d,
            texttemplate="%{text}"
        ))

        # ---------------------------
        # Abwechselnde Tageshintergründe
        # ---------------------------
        shapes = []
        for d in range(D):
            # Jede Taggruppe (3 Spalten = FS, SS, NS)
            x0 = d * S - 0.5
            x1 = (d + 1) * S - 0.5

            # Nur jeden zweiten Tag einfärben
            if d % 2 == 0:
                shapes.append(dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(230,230,230,0.6)",  # leichtes Grau
                    line_width=0,
                    layer="below"
                ))

        fig.update_layout(
            title="Dienstplan Heatmap (grün = Schicht zugewiesen)",
            xaxis_nticks=21,
            yaxis_autorange='reversed',
            width=1000,
            height=400,
            shapes=shapes,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)


        # ---------------------------
        # Barplot: Gesamtanzahl der Schichten pro Krankenschwester
        # ---------------------------
        st.write("### Gesamtanzahl der Schichten pro Krankenschwester")
        
        # Berechne die Gesamtanzahl der Schichten pro Krankenschwester
        shifts_per_nurse = np.nansum(best_schedule, axis=(1,2))  # Summe über Tage & Schichten

        # DataFrame für Plotly
        df_shifts = pd.DataFrame({
            "Krankenschwester": [f"KS{i+1}" for i in range(N)],
            "Anzahl Schichten": shifts_per_nurse
        })

        # Vertikaler Barplot
        fig_bar = px.bar(
            df_shifts,
            x="Krankenschwester",
            y="Anzahl Schichten",
            text="Anzahl Schichten",
            color="Anzahl Schichten",
            color_continuous_scale="Greens"
        )

        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            title="Gesamtanzahl der Schichten pro Krankenschwester",
            yaxis_title="Anzahl Schichten",
            xaxis_title="Krankenschwester",
            yaxis=dict(tick0=0, dtick=1),
            width=800,
            height=400
        )

        st.plotly_chart(fig_bar, use_container_width=True)


        # ---------------------------
        # 3D Pheromon Animation
        # ---------------------------
        x, y, z, c, itr_list = [], [], [], [], []
        for itr, tau_m in enumerate(tau_history):
            for n_ in range(N):
                for d_ in range(D):
                    for s_ in range(S):
                        x.append(n_)
                        y.append(d_)
                        z.append(s_)
                        c.append(tau_m[n_,d_,s_])
                        itr_list.append(itr+1)
        df = pd.DataFrame({"Nurse":x,"Day":y,"Shift":z,"Pheromone":c,"Iteration":itr_list})
        fig = px.scatter_3d(
            df, x="Nurse", y="Day", z="Shift", color="Pheromone",
            animation_frame="Iteration", color_continuous_scale="Viridis",
            range_color=[df["Pheromone"].min(), df["Pheromone"].max()]
        )
        fig.update_layout(scene=dict(xaxis_title='Nurse',yaxis_title='Day',zaxis_title='Shift'),
                          title="Pheromone Entwicklung über Iterationen")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Tab 4: Discussion
# ---------------------------
with tabs[3]:
    st.markdown("## Diskussion")
    st.markdown(
        """
    - Analyse der Lösungsqualität und Effizienz  
    - Vergleich erwarteter vs. unerwarteter Ergebnisse  
    - Limitationen der Implementierung  
    - Mögliche Verbesserungen (z. B. dynamische Heuristik, adaptive Parameterwahl)  
    - Komplexität bei größeren Problemgrößen
"""
    )


